import os
import numpy as np
import pyglet
import logging 
import logfire
from wmk.messenger import Messenger
from wmk.player import Player

def setup_logging():
    # Configure Logfire to send logs to the Logfire service.
    # This is optional, but very helpful for debugging deployed applications.
    # logfire.configure(token='YOUR_LOGFIRE_TOKEN')
    # logging.basicConfig(handlers=[logfire.LogfireLoggingHandler()])

    # Ensure we display INFO logs.
    logging.basicConfig(level=logging.INFO)
    # Create our standard logger.
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    return logger

logger = setup_logging()

# Pre-create the frames we will display. 
red_frame = np.zeros((512, 512, 3), dtype=np.uint8)
red_frame[:,:] = [200, 99, 99]
green_frame = np.zeros((512, 512, 3), dtype=np.uint8)
green_frame[:,:] = [99, 200, 99] 
blue_frame = np.zeros((512, 512, 3), dtype=np.uint8)
blue_frame[:,:] = [99, 99, 200]
white_frame = np.zeros((512, 512, 3), dtype=np.uint8)
white_frame[:,:] = [200, 200, 200]

def init():
    global player, is_connected

    is_connected = False
    logger.info('Init: Starting')
    init_messenger()
    player = Player(
                frame_generator=generate_frames,
                fps_max=60,
                fps_display=True,
                width=800,
                height=600,
                caption="WMK Demo"
    )
    player.run()


def generate_frames(player: Player, dt: float):
    global is_connected

    # Check if G key is in the keys_pressed set
    # We use player.keys_pressed rather than player.keyboard_state,
    # because if the frame generator is too slow, a key might be 
    # pressed and released before the event handler is called.
    # In other cases, it might be better to use player.keyboard_state.
    # g_key_pressed = pyglet.window.key.G in player.keys_pressed
    a_key_pressed = pyglet.window.key.A in player.keys_pressed
    d_key_pressed = player.keyboard_state[pyglet.window.key.D]
    w_key_pressed = player.keyboard_state[pyglet.window.key.W]
    
    if a_key_pressed or d_key_pressed:
        return green_frame
    elif w_key_pressed:
        return white_frame
    elif is_connected:
        return blue_frame
    else:
        return red_frame


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


if __name__ == "__main__":
    init()