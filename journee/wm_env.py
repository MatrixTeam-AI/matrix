import argparse
from typing import Literal, Optional
from dataclasses import dataclass
import numpy as np
import random
import sys, os
sys.path.insert(0, os.path.abspath('.'))
import torch
from stage4_ray.pipeline_cogvideox_interactive import CogVideoXInteractiveStreamingPipeline
from stage4_ray.pipeline_cogvideox_interactive import CogVideoXTransformer3DModel
from stage4_ray.pipeline_cogvideox_interactive import LCMSwinScheduler

from stage4_ray.pipeline_cogvideox_interactive import CogVideoXPipelineOutput
from stage4_ray.pipeline_cogvideox_interactive import CogVideoXLoraLoaderMixin
from stage4_ray.pipeline_cogvideox_interactive import AutoencoderKLCogVideoX
from stage4_ray.pipeline_cogvideox_interactive import CogVideoXTransformer3DModel
from stage4_ray.pipeline_cogvideox_interactive import (
    LCMSwinScheduler,
    CogVideoXDPMScheduler,
    CogVideoXSwinDPMScheduler,
    expand_timesteps_with_group,
)
from stage4_ray.pipeline_cogvideox_interactive import CONTROL_SIGNAL_TO_PROMPT

from diffusers.utils import export_to_video, load_image, load_video

import decord
import PIL.Image
import datetime

CONTROL_FILE = "./COMMAND.txt"

def read_control_signal(filepath=CONTROL_FILE):
    try:
        with open(filepath, 'r') as f:
            return f.read().strip().upper()
    except FileNotFoundError:
        return None

@dataclass
class InteractiveGenerationArgs:
    prompt: str
    model_path: str
    video_path: str
    lora_path: Optional[str] = None
    lora_rank: int = 256
    output_path: str = "./output.mp4"
    num_inference_steps: int = 4
    num_frames: int = 17
    width: int = 720
    height: int = 480
    fps: int = 16
    num_videos_per_prompt: int = 1
    dtype: str = "bfloat16"
    seed: int = 42
    gpu_id: int = 0
    use_dynamic_cfg: bool = False
    do_classifier_free_guidance: bool = False
    
    # swin arguments
    num_noise_groups: int = 4
    init_video_clip_frame: int = 17

    # lcm arguments
    original_inference_steps: int = 40
    lcm_multiplier: int = 1
    
    # gym arguments
    max_iteractions: int = 100000
    
    def __post_init__(self):
        if self.dtype == "float16":
            self.dtype = torch.float16
        elif self.dtype == "bfloat16":
            self.dtype = torch.bfloat16
        else:
            raise ValueError(f"Unsupported dtype: {self.dtype}")

def seed_everything(seed=42):
    """
    Set the random seed for all libraries to a fixed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def prepare_wm_env_init_args(wm_gen_config):
    # init_video should be pillow list.
    video_reader = decord.VideoReader(wm_gen_config.video_path)
    video_num_frames = wm_gen_config.num_frames
    video_fps = video_reader.get_avg_fps()
    sampling_interval = video_fps/wm_gen_config.fps
    frame_indices = np.round(np.arange(0, video_num_frames, sampling_interval)).astype(int).tolist()
    frame_indices = frame_indices[:wm_gen_config.init_video_clip_frame]
    video = video_reader.get_batch(frame_indices).asnumpy()
    video = [PIL.Image.fromarray(frame) for frame in video]

    # Generate the video frames based on the prompt.
    wm_gen_config.num_frames = len(video)
    init_args = {
        "prompt": wm_gen_config.prompt,
        "num_videos_per_prompt": wm_gen_config.num_videos_per_prompt,
        "num_inference_steps": wm_gen_config.num_inference_steps,
        "height": wm_gen_config.height,
        "width": wm_gen_config.width,
        "num_frames": wm_gen_config.num_frames,
        "use_dynamic_cfg": wm_gen_config.use_dynamic_cfg,
        "generator": torch.Generator().manual_seed(wm_gen_config.seed),
        "control_signal": None,
        "init_video": video,
        "num_noise_groups": wm_gen_config.num_noise_groups,
        "original_inference_steps": wm_gen_config.original_inference_steps,
        "lcm_multiplier": wm_gen_config.lcm_multiplier,
        "do_classifier_free_guidance": False,
    }
    return init_args

@torch.no_grad()
def load_matrix_interactive_pipe(wm_gen_config, disable_progress_bar=False):
    transformer = CogVideoXTransformer3DModel.from_pretrained(
            os.path.join(wm_gen_config.model_path, "transformer"),
            torch_dtype=wm_gen_config.dtype,
        )
    # NOTE: `keep_cache` feature conflicts with the tiling/slicing feature
    vae = AutoencoderKLCogVideoX.from_pretrained(os.path.join(wm_gen_config.model_path, 'vae'), 
                                                    torch_dtype=wm_gen_config.dtype)
    # vae.to(0)
    # v = vae.encode(torch.randn((1, 3, 17, 480, 720), dtype=torch.float16).to(vae.device))
    pipe = CogVideoXInteractiveStreamingPipeline.from_pretrained(wm_gen_config.model_path, 
                                                        vae=vae, transformer=transformer, 
                                                        torch_dtype=wm_gen_config.dtype)
    pipe.scheduler = LCMSwinScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=disable_progress_bar) 
    if wm_gen_config.lora_path:  # If you're using with lora, add this code
        pipe.load_lora_weights(wm_gen_config.lora_path, 
                               weight_name="pytorch_lora_weights.safetensors")
        pipe.fuse_lora(components=["transformer"],)  # lora_scale=1 / lora_rank  # It seems that there are some issues here, removed.
    pipe.to(wm_gen_config.gpu_id)  # pipe._execution_device from cpu --> cuda:0
    pipe.wm_gen_config = wm_gen_config
    return pipe
    

def test_interactive():
    interactive_gen_config = InteractiveGenerationArgs(
        prompt="On a lush green meadow, a white car is driving. From an overhead panoramic shot, \
                this car is adorned with blue and red stripes on its body, and it has a black spoiler at the rear. \
                The camera follows the car as it moves through a field of golden wheat, surrounded by green grass and trees. \
                In the distance, a river and some hills can be seen, with a cloudless blue sky above.",
        model_path="/home/andy/matrix_stage4_ckpt",
        video_path="/home/andy/matrix/base_video.mp4",
    )  # natural domain randomization support :)
    seed_everything(interactive_gen_config.seed)
    
    debug_clip_output_dir = "./debug_clip_output"
    os.makedirs(debug_clip_output_dir, exist_ok=True)
    
    wm_interactive_pipe = load_matrix_interactive_pipe(interactive_gen_config)
    
    wm_init_args = prepare_wm_env_init_args(interactive_gen_config)
    wm_interactive_pipe.env_init(**wm_init_args)
    

def generate_base_video():
    pass

if __name__ == "__main__":
    test_interactive()