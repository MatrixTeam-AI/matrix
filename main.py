from journee.utils.log_utils import redirect_stdout_err_to_logger, logger
redirect_stdout_err_to_logger(logger)
import os
import time

import numpy as np
import pyglet
from wmk.messenger import Messenger
from wmk.player import Player

#============ Interfaces with the backend model ===========#
#TODO: implement these interfaces and import them
# class DummyQueue:
#     def get(self): # block
#         return None
#     def put(self, control): # non-block
#         pass
# class DummyModel:
#     def start(self):
#         r"""Start video generation.
#         By default, if no control is given, the signal will be set to 'D'.
#         """
#         pass
#     def init_generation(self):
#         r"""prepare the warm-up video"""
#         pass
# def init_model():
#     return DummyModel(), DummyQueue(), DummyQueue() # model, frame_queue, control_queue
#============ Interfaces with the backend model ===========#
from journee.interface import init_model, passed_times_dict_to_str
import ray
current_state_var = ray.get_actor(namespace='matrix', name="current_state")

def random_control_signal_generator(
        seed, 
        repeat_lens=[2, 4, 4],
        signal_choices=['D', 'DR', 'DL'],
        change_prob_increment=0.2,
    ):
        if not signal_choices or not repeat_lens \
            or len(repeat_lens) != len(signal_choices):
            raise ValueError("Invalid parameters")
        rng = np.random.default_rng(seed)
        current_repeat = 0
        current_idx = 0
        change_prob = change_prob_increment
        while True:
            if current_repeat >= repeat_lens[current_idx]:
                if change_prob >= 1 or rng.uniform(0, 1) < change_prob:
                    if current_idx == 0:
                        current_idx_choices = [j for j in range(1, len(signal_choices))]
                        current_idx = rng.choice(current_idx_choices)
                    else:
                        current_idx = 0
                    current_repeat = 1
                    change_prob = change_prob_increment
                else:
                    current_repeat += 1
                    change_prob = min(1, change_prob + change_prob_increment)
            else:
                current_repeat += 1
            yield signal_choices[current_idx]

WIDTH = 720
HEIGHT = 480
VIDEO_FPS = 1000
CONTROL_FPS = 1000
DISPLAY_FPS = 18 # determine the frequency to update frame and control signals
CONTROL_GENERATOR = random_control_signal_generator(0)

# Pre-create the frames for default cases. 
RED_FRAME = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
RED_FRAME[:,:] = [200, 99, 99]
GREEN_FRAME = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
GREEN_FRAME[:,:] = [99, 200, 99] 
BLUE_FRAME = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
BLUE_FRAME[:,:] = [99, 99, 200]
WHITE_FRAME = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
WHITE_FRAME[:,:] = [200, 200, 200]

def init_messenger():
    global is_connected

    # The messenger functionality is currently only supported on Unix systems
    # and intended for use in the Ojin cloud.
    # Knowing when a new user connects and disconnects is useful for:
    # - Resetting the user experience.
    # - Pausing the generation of frames while there are no users, this 
    #   lets cleanup processes such as garbage collection run. 
    if os.name != 'nt':
        logger.info("Init: Creating messenger")
        messenger = Messenger(
            connection_retry_interval=5,
            connection_timeout=120,
        )
        logger.info("Init: Setting messenger handlers")
        messenger.register_event_type("PlayerConnected")
        messenger.register_event_type("PlayerDisconnected")

        # Use async event handlers with messenger
        def on_player_connected(msg):
            global is_connected
            logger.info("Player Connected")
            is_connected = True

        def on_player_disconnected(msg):
            global is_connected
            is_connected = False
            logger.info("Player Disconnected")

        messenger.set_handlers(
            PlayerConnected=on_player_connected,
            PlayerDisconnected=on_player_disconnected,
        )

        logger.info("Init: Starting messenger")
        messenger.start()
        logger.info("Init: Messenger started")
    else:
        logger.info("Init: Running on Windows, skipping Messenger init...")

class FrameManager:
    r""" Determine the frame to display, such that the video is played in a normal speed.
    For example, if the video should be played at 16 fps, but the frame is updated at 64 fps,
    then each effective video frame should last for around 64 / 16  = 4 display frames.
    In practice, the time interval between two updates of the display frame is not constant and will fluctuate.
    So in order to play the generated video in a normal speed, we need to properly manage the frame update.
    """
    def __init__(
        self,
        frame_queue,
        video_fps: float,
        display_fps: int,
    ):
        r"""
        frame_queue: a queue that contains the generated frames.
        video_fps: the fps of the generated video. 
        display_fps: the fps of the front-end display.
        """
        self.frame_queue = frame_queue
        self.video_fps = video_fps
        self.display_fps = display_fps
        
        self.last_frame = None
        self.last_frame_time = 100. # a big enough number to make sure the first frame is displayed
        self.video_dt = 1 / self.video_fps

    def get(self, dt, if_wait_nonempty=False):
        """
        dt: Time elapsed since last update in seconds
        """
        # a simple implementation
        frame = self.last_frame
        self.last_frame_time += dt
        # logger.info(
        #     f"[main.FrameManager.get]"
        #     f" dt: {dt:.3f}s, last_frame_time: {self.last_frame_time:.3f}s"
        #     f" {self.frame_queue.size()=}"
        # )
        if (
            (self.last_frame is None or self.last_frame_time >= self.video_dt)
            and (if_wait_nonempty or not self.frame_queue.empty())
        ):
            # logger.info(f"[main.FrameManager.get] Getting frame...")
            frame, passed_times = self.frame_queue.get()
            passed_times_str = passed_times_dict_to_str(passed_times)
            global frame_counter
            # if frame_counter % 1 == 0:
            #     logger.info(f"[main.FrameManager.get] Got frame! passed_times:\n{passed_times_str}")
            frame = frame[::-1] # flip the H dimension for wmk
            self.last_frame = frame
            self.last_frame_time = dt
        return frame

class ControlManager:
    r"""Determine whether to update the control queue.
    """
    def __init__(
        self,
        control_queue,
        control_fps: float,
        display_fps: int,
    ):
        self.control_queue = control_queue
        self.control_fps = control_fps
        self.display_fps = display_fps

        self.last_control_time = 100. # a big enough number to make sure the first control is entered.
        self.control_dt = 1 / self.control_fps

    def put(self, control, dt, check_full=True):
        """
        control: current control
        dt: Time elapsed since last update in seconds
        """
        # a simple implementation
        self.last_control_time += dt
        # logger.info(
        #     f"[main.ControlManager.put]"
        #     f" dt: {dt:.3f}s, last_control_time: {self.last_control_time:.3f}s"
        #     f" {self.control_queue.size()=}"
        # )
        if (
            self.last_control_time >= self.control_dt
            and not check_full or not self.control_queue.full()
        ):
            # logger.info(f"[main.ControlManager.put] Putting control...")
            self.control_queue.put(control) #TODO: may need to modify the interface of `control_queue`
            self.last_control_time = dt

def generate_frames(player: Player, dt: float):
    global is_connected, frame_manager, control_manager, frame_counter
    global CONTROL_GENERATOR

    # Collect control signals A/W/D.
    # 2 ways to check if a key is pressed:
    # 1) a_key_pressed = pyglet.window.key.A in player.keys_pressed
    # 2) a_key_pressed = player.keyboard_state[pyglet.window.key.A]
    # We use player.keys_pressed rather than player.keyboard_state,
    # because if the frame generator is too slow, a key might be 
    # pressed and released before the event handler is called.
    # In other cases, it might be better to use player.keyboard_state.

    a_key_pressed = pyglet.window.key.A in player.keys_pressed
    d_key_pressed = pyglet.window.key.D in player.keys_pressed
    w_key_pressed = pyglet.window.key.W in player.keys_pressed
    # Get control and update control queue
    # control = 'D'   # Default. Move forward even if no key is pressed.
    control = next(CONTROL_GENERATOR)
    if a_key_pressed:
        control = 'DL'
    elif d_key_pressed:
        control = 'DR'
    elif w_key_pressed:
        control = 'D'
    # print(f"putting control signal {control} to the queue")
    if ray.get(current_state_var.get.remote()) == "RUN":
        control_manager.put(control, dt, check_full=False)    # non-blocking
    if frame_counter == 0:
        logger.info(f"[main.generate_frames] send control {control} @ {frame_counter=}")
    # print("Waiting for frames")
    # Get the frame to display
    if ray.get(current_state_var.get.remote()) == "RUN":
        frame = frame_manager.get(
            dt,
            if_wait_nonempty=True, # blocking
        )
    else:
        frame = None
    if frame_counter == 0:
        logger.info(f"[main.generate_frames] get frame @ {frame_counter=}")
    
    # when no frame is available, we display a color frame
    if frame is None:
        frame = BLUE_FRAME
    frame_counter += 1
    return frame
    
def main():
    global player, is_connected, frame_manager, control_manager, frame_counter
    is_connected = False
    frame_counter = 0
    logger.info('Init: Starting')
    # init_messenger()

    # Backend: init model and start the video generation
    model, frame_queue, control_queue = init_model() # block
    frame_manager = FrameManager(
        frame_queue=frame_queue,
        video_fps=VIDEO_FPS,
        display_fps=DISPLAY_FPS,
    )
    control_manager = ControlManager(
        control_queue=control_queue,
        control_fps=CONTROL_FPS,
        display_fps=DISPLAY_FPS,
    )
    model.init_generation() # block
    model.start()           # non-block

    # Frontend: init Player - the game window
    player = Player(
        frame_generator=generate_frames,
        fps_max=DISPLAY_FPS,
        fps_display=True,
        width=720,
        height=480,
        caption="The Matrix"
    )
    logger.info('player.run')
    player.run()

if __name__ == "__main__":
    main()