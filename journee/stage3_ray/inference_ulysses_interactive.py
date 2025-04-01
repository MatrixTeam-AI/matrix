# This file is modified from: https://github.com/xdit-project/xDiT/blob/main/examples/cogvideox_usp_example.py
import functools
from typing import List, Optional, Tuple, Union
import argparse

import gc
import logging
import time
import torch
import numpy as np
import decord
import PIL.Image

from diffusers import DiffusionPipeline

from xfuser import xFuserArgs
from xfuser.config import FlexibleArgumentParser
from xfuser.core.distributed import (
    get_world_group,
    get_runtime_state,
    get_classifier_free_guidance_world_size,
    get_classifier_free_guidance_rank,
    get_cfg_group,
    get_sequence_parallel_world_size,
    get_sequence_parallel_rank,
    get_sp_group,
    is_dp_last_group,
    initialize_runtime_state,
    get_pipeline_parallel_world_size,
)

from diffusers.utils import export_to_video

# from xfuser.model_executor.layers.attention_processor import xFuserCogVideoXAttnProcessor2_0

import os
import sys
sys.path.insert(0, '/'.join(os.path.realpath(__file__).split('/')[:-2]))
from pipeline_cogvideox_interactive  import CogVideoXInteractiveStreamingPipeline
from stage3.cogvideox.transformer import CogVideoXTransformer3DModel
from stage3.cogvideox.autoencoder import AutoencoderKLCogVideoX
from stage3.cogvideox.scheduler import CogVideoXSwinDPMScheduler
from stage3.cogvideox.parallel_vae_utils import VAEParallelState

def generate_random_control_signal(
        length, seed, repeat_lens=[2, 2, 2], signal_choices=['D', 'DR', 'DL'],
        change_prob_increment=0.2,
    ):
        if not signal_choices or not repeat_lens \
            or len(repeat_lens) != len(signal_choices) \
            or length < 1:
            raise ValueError("Invalid parameters")
        rng = np.random.default_rng(seed)
        result = []
        current_repeat = 0
        current_idx = 0
        change_prob = change_prob_increment
        for i in range(length):
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
            result.append(signal_choices[current_idx])
        return ','.join(result)
        
def init_pipeline(
    model_path: str,
    local_rank: int,
    lora_path: str = None,
    lora_rank: int = 128,
    transformer_path: str = "",
    dtype: torch.dtype = torch.bfloat16,
    enable_sequential_cpu_offload: bool = False,
    enable_model_cpu_offload: bool = False,
    enable_tiling: bool = True,
    enable_slicing: bool = True,
    parallel_decoding_idx: int = -1,
):
    transformer = CogVideoXTransformer3DModel.from_pretrained(
        transformer_path or os.path.join(model_path, "transformer"),
        torch_dtype=dtype,
        low_cpu_mem_usage=False
    )
    vae = AutoencoderKLCogVideoX.from_pretrained(
        os.path.join(model_path, "vae"),
        torch_dtype=dtype,
    )
    scheduler = CogVideoXSwinDPMScheduler.from_config(
        os.path.join(model_path, "scheduler"), 
        timestep_spacing="trailing"
    )
    pipe = CogVideoXInteractiveStreamingPipeline.from_pretrained(
        model_path, 
        transformer=transformer, 
        vae=vae,
        scheduler=scheduler, 
        torch_dtype=dtype
    )

    # If you're using with lora, add this code
    if lora_path:
        pipe.load_lora_weights(lora_path, weight_name="pytorch_lora_weights.safetensors")
        pipe.fuse_lora(components=["transformer"],
            # lora_scale=1 / lora_rank  # It seems that there are some issues here, removed.
        )

    if enable_sequential_cpu_offload:
        pipe.enable_sequential_cpu_offload(gpu_id=local_rank)
        logging.info(f"rank {local_rank} sequential CPU offload enabled")
    elif enable_model_cpu_offload:
        pipe.enable_model_cpu_offload(gpu_id=local_rank)
        logging.info(f"rank {local_rank} model CPU offload enabled")
    else:
        device = torch.device(f"cuda:{local_rank}")
        pipe = pipe.to(device)

    # pipe.enable_sequential_cpu_offload()
    if enable_tiling:
        pipe.vae.enable_tiling()

    if enable_slicing:
        pipe.vae.enable_slicing()

    if parallel_decoding_idx > -1:
        pipe.vae.enable_parallel_decoding(parallel_decoding_idx)

    return pipe

def generate_video(
    prompt: str,
    pipe: DiffusionPipeline, 
    num_frames: int = 81,
    width: int = 1360,
    height: int = 768,
    image_or_video_path: str = "",
    num_inference_steps: int = 50,
    guidance_scale: float = 6.0,
    num_videos_per_prompt: int = 1,
    seed: int = 42,
    fps: int = 8,
    num_noise_groups: int=4,
    num_sample_groups: int = 20,
    init_video_clip_frame: int = 65,
):
    """
    Generates a video based on the given prompt and saves it to the specified path.

    Parameters:
    - prompt (str): The description of the video to be generated.
    - model_path (str): The path of the pre-trained model to be used.
    - lora_path (str): The path of the LoRA weights to be used.
    - lora_rank (int): The rank of the LoRA weights.
    - output_path (str): The path where the generated video will be saved.
    - num_inference_steps (int): Number of steps for the inference process. More steps can result in better quality.
    - num_frames (int): Number of frames to generate. CogVideoX1.0 generates 49 frames for 6 seconds at 8 fps, while CogVideoX1.5 produces either 81 or 161 frames, corresponding to 5 seconds or 10 seconds at 16 fps.
    - width (int): The width of the generated video, applicable only for CogVideoX1.5-5B-I2V
    - height (int): The height of the generated video, applicable only for CogVideoX1.5-5B-I2V
    - guidance_scale (float): The scale for classifier-free guidance. Higher values can lead to better alignment with the prompt.
    - num_videos_per_prompt (int): Number of videos to generate per prompt.
    - dtype (torch.dtype): The data type for computation (default is torch.bfloat16).
    - seed (int): The seed for reproducibility.
    - fps (int): The frames per second for the generated video.
    """
    # Init_video should be pillow list.
    video_reader = decord.VideoReader(image_or_video_path)
    video_num_frames = len(video_reader)
    video_fps = video_reader.get_avg_fps()
    sampling_interval = video_fps/fps
    frame_indices = np.round(np.arange(0, video_num_frames, sampling_interval)).astype(int).tolist()
    frame_indices = frame_indices[:init_video_clip_frame]
    video = video_reader.get_batch(frame_indices).asnumpy()
    video = [PIL.Image.fromarray(frame) for frame in video]

    # 4. Generate the video frames based on the prompt.
    num_frames = len(video)
    print(f"{len(video)=}")
    
        
    with torch.no_grad():
        # interactive video generation
        pipe(
            prompt=prompt,
            num_videos_per_prompt=num_videos_per_prompt,
            num_inference_steps=num_inference_steps,
            height=height,
            width=width,
            num_frames=num_frames,
            use_dynamic_cfg=False,  # This is used for DPM scheduler, for DDIM scheduler, it should be False
            guidance_scale=guidance_scale,
            generator=torch.Generator(device='cuda').manual_seed(seed),
            init_video=video,
            num_noise_groups=num_noise_groups,
            num_sample_groups=num_sample_groups,
        )

def add_argument_overridable(parser, *args, **kwargs):
    # collection existing options
    existing_option_strings = set()
    for action in parser._actions:
        existing_option_strings.update(action.option_strings)
    
    # check existing args
    for arg in args:
        if arg in existing_option_strings:
            remove_argument(parser, arg)

    # add new args
    parser.add_argument(*args, **kwargs)

def remove_argument(parser, option_string):
    if option_string in parser._option_string_actions:
        parser._option_string_actions.pop(option_string)
    action_to_remove = None
    for action in parser._actions:
        if option_string in action.option_strings:
            action_to_remove = action
            break
    if action_to_remove:
        parser._actions.remove(action_to_remove)
        for group in parser._mutually_exclusive_groups:
            if action_to_remove in group._group_actions:
                group._group_actions.remove(action_to_remove)

def main():
    parser = FlexibleArgumentParser(description="xFuser Arguments")
    parser = xFuserArgs.add_cli_args(parser)
    add_argument_overridable(parser, "--prompt", type=str, help="The description of the video to be generated")
    add_argument_overridable(parser, "--model_path", type=str, help="Path of the pre-trained model use")
    add_argument_overridable(parser, "--transformer_path", type=str, default="", help="Transformer save path in stage3 training.")
    add_argument_overridable(parser, "--image_or_video_path", type=str, help="The path of the image to be used as the background of the video.")
    add_argument_overridable(parser, "--lora_path", type=str, default=None, help="The path of the LoRA weights to be used")
    add_argument_overridable(parser, "--lora_rank", type=int, default=256, help="The rank of the LoRA weights")
    add_argument_overridable(parser, "--output_path", type=str, default="./output.mp4", help="The path save generated video")
    add_argument_overridable(parser, "--guidance_scale", type=float, default=6.0, help="The scale for classifier-free guidance")
    add_argument_overridable(parser, "--num_inference_steps", type=int, default=4, help="Inference steps")
    add_argument_overridable(parser, "--num_frames", type=int, default=41, help="NOT USED HERE")
    add_argument_overridable(parser, "--width", type=int, default=720, help="Number of steps for the inference process")
    add_argument_overridable(parser, "--height", type=int, default=480, help="Number of steps for the inference process")
    add_argument_overridable(parser, "--fps", type=int, default=16, help="Number of steps for the inference process")
    add_argument_overridable(parser, "--num_videos_per_prompt", type=int, default=1, help="Number of videos to generate per prompt")
    add_argument_overridable(parser, "--dtype", type=str, default="bfloat16", help="The data type for computation")
    add_argument_overridable(parser, "--seed", type=int, default=42, help="The seed for reproducibility")
    # swin arguments
    add_argument_overridable(parser, "--num_noise_groups", type=int, default=4, help="Number of noise groups")
    add_argument_overridable(parser, "--num_sample_groups", type=int, default=8, help="Number of sampled videos groups")
    add_argument_overridable(parser, "--init_video_clip_frame", type=int, default=17, help="Frame number of init_video to be clipped, should be 4n+1")
    # parallel arguments
    add_argument_overridable(parser, "--split_text_embed_in_sp", type=str, default="true", choices=["true", "false", "auto"], help="Whether to split text embed `encoder_hidden_states` for sequence parallel.")
    add_argument_overridable(parser, "--parallel_decoding_idx", type=int, default=-1, choices=[-1, 0, 1, 2, 3], help="Upblock index in VAE.decoder to enable parallel decoding. -1 means disabling parallel decoding.")
    args = parser.parse_args()

    engine_args = xFuserArgs.from_cli_args(args)
    engine_config, input_config = engine_args.create_config()
    local_rank = get_world_group().local_rank

    assert engine_args.pipefusion_parallel_degree == 1, "This script does not support PipeFusion."
    assert engine_args.use_parallel_vae is False, "parallel VAE not implemented for CogVideo"
    assert engine_config.runtime_config.use_torch_compile is False, "`use_torch_compile` is not supported yet."

    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    assert not args.enable_tiling and not args.enable_slicing, "Tiling and slicing are not supported yet."
    pipe = init_pipeline(
        model_path=args.model_path,
        transformer_path=args.transformer_path,
        local_rank=local_rank,
        lora_path=args.lora_path,
        lora_rank=args.lora_rank,
        dtype=dtype,
        enable_sequential_cpu_offload=args.enable_sequential_cpu_offload,
        enable_model_cpu_offload=args.enable_model_cpu_offload,
        enable_tiling=args.enable_tiling,
        enable_slicing=args.enable_slicing,
        parallel_decoding_idx=args.parallel_decoding_idx,
    )
    pipe.set_progress_bar_config(disable=True)  # by default, disable progress bar during the denoising process
    parameter_peak_memory = torch.cuda.max_memory_allocated(device=f"cuda:{local_rank}")

    initialize_runtime_state(pipe, engine_config)  # set up `xfuser.core.distributed.runtime_state.DiTRuntimeState`
    if args.parallel_decoding_idx >= 0:
        VAEParallelState.initialize(vae_group=get_world_group().device_group)
    split_text_embed_in_sp = {
        "true": True,
        "false": False,
        "auto": None,
    }[args.split_text_embed_in_sp]
    get_runtime_state().set_video_input_parameters(  # get_runtime_state returns a `xfuser.core.distributed.runtime_state.DiTRuntimeState`
        height=input_config.height,
        width=input_config.width,
        num_frames=input_config.num_frames,
        batch_size=1,
        num_inference_steps=input_config.num_inference_steps,
        split_text_embed_in_sp=split_text_embed_in_sp,
    )
    
    # if engine_config.runtime_config.use_torch_compile:
    #     torch._inductor.config.reorder_for_compute_comm_overlap = True
    #     pipe.transformer = torch.compile(pipe.transformer, mode="max-autotune-no-cudagraphs")

    #     # one step to warmup the torch compiler
    #     output = pipe(
    #         height=input_config.height,
    #         width=input_config.width,
    #         num_frames=input_config.num_frames,
    #         prompt=input_config.prompt,
    #         num_inference_steps=1,
    #         generator=torch.Generator(device="cuda").manual_seed(input_config.seed),
    #     ).frames[0]

    torch.cuda.reset_peak_memory_stats()

    start_time = time.time()
    generate_video(
        prompt=args.prompt,
        pipe=pipe,
        num_frames=args.num_frames,
        width=args.width,
        height=args.height,
        image_or_video_path=args.image_or_video_path,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        num_videos_per_prompt=args.num_videos_per_prompt,
        seed=args.seed,
        fps=args.fps,
        num_sample_groups=args.num_sample_groups,
        num_noise_groups=args.num_noise_groups,
        init_video_clip_frame=args.init_video_clip_frame,
    )
    end_time = time.time()
    elapsed_time = end_time - start_time
    peak_memory = torch.cuda.max_memory_allocated(device=f"cuda:{local_rank}")


    if get_world_group().rank == get_world_group().world_size - 1:
        print(f"epoch time: {elapsed_time:.2f} sec, parameter memory: {parameter_peak_memory/1e9:.2f} GB, memory: {peak_memory/1e9} GB")
    get_runtime_state().destory_distributed_env()


if __name__ == "__main__":
    main()