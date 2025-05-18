# This file is modified from: https://github.com/xdit-project/xDiT/blob/main/examples/cogvideox_usp_example.py
import os
import sys
sys.path.insert(0, '/'.join(os.path.realpath(__file__).split('/')[:-3]))
from journee.utils.log_utils import redirect_stdout_err_to_logger, logger
# redirect_stdout_err_to_logger(logger)
FILE_NAME = os.path.splitext(os.path.basename(__file__))[0]
print(f"[{FILE_NAME}] {__file__} loaded")

import functools
from typing import List, Optional, Tuple, Union
import argparse
import gc
import logging
import time

import torch
import numpy as np

from diffusers import DiffusionPipeline
from diffusers.utils import export_to_video

from pipeline_cogvideox_interactive  import CogVideoXInteractiveStreamingPipeline
from stage4.cogvideox.transformer import CogVideoXTransformer3DModel
from stage4.cogvideox.autoencoder import AutoencoderKLCogVideoX
from stage4.cogvideox.scheduler import LCMSwinScheduler
from stage4.cogvideox.parallel_vae_utils import VAEParallelState

from stage2.inference import generate_video as base_gen

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
# from xfuser.model_executor.layers.attention_processor import xFuserCogVideoXAttnProcessor2_0
from journee.stage4_ray.config import GenerationConfig, GenerationStepConfig, DecodeConfig

def init_pipeline(
    engine_config,
    input_config,
    model_path: str,
    local_rank: int,
    lora_path: str = None,
    lora_rank: int = 128,
    dtype: torch.dtype = torch.float16,
    enable_sequential_cpu_offload: bool = False,
    enable_model_cpu_offload: bool = False,
    enable_tiling: bool = True,
    enable_slicing: bool = True,
    parallel_decoding_idx: int = -1,
    split_text_embed_in_sp: Optional[bool] = None,
):
    transformer, transformer_loading_info = CogVideoXTransformer3DModel.from_pretrained(
        os.path.join(model_path, "transformer"),
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        output_loading_info=True,
    )
    print(f"[{FILE_NAME}.init_pipeline] {transformer_loading_info=}")
    vae = AutoencoderKLCogVideoX.from_pretrained(
        os.path.join(model_path, "vae"),
        torch_dtype=dtype,
    )
    scheduler = LCMSwinScheduler.from_config(
        os.path.join(model_path, "scheduler"), 
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

    pipe.set_progress_bar_config(disable=True)  # by default, disable progress bar during the denoising process

    initialize_runtime_state(pipe, engine_config)  # set up `xfuser.core.distributed.runtime_state.DiTRuntimeState`
    
    if parallel_decoding_idx > -1:
        VAEParallelState.initialize(vae_group=get_world_group().device_group)
    
    get_runtime_state().set_video_input_parameters(  # get_runtime_state returns a `xfuser.core.distributed.runtime_state.DiTRuntimeState`
        height=input_config.height,
        width=input_config.width,
        num_frames=input_config.num_frames,  # default to 49 in xFuser
        batch_size=1,
        num_inference_steps=input_config.num_inference_steps,
        split_text_embed_in_sp=split_text_embed_in_sp,
    )    
    return pipe

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

@torch.no_grad()
def main():
    deault_prompt = "The video shows a white car driving on a country road on a sunny day. The car comes from \
              the back of the scene, moving forward along the road, with open fields and distant hills \
              surrounding it. As the car moves, the vegetation on both sides of the road and distant buildings can be seen. \
              The entire video records the car's journey through the natural environment using a follow-shot technique."
    parser = FlexibleArgumentParser(description="xFuser Arguments")
    parser = xFuserArgs.add_cli_args(parser)
    add_argument_overridable(parser, "--prompt", type=str, default=deault_prompt, help="The description of the video to be generated")
    add_argument_overridable(parser, "--model_path", type=str, help="Path of the pre-trained model use")
    add_argument_overridable(parser, "--video_cache_dir", type=str, help="The path of the base video cache directory")
    add_argument_overridable(parser, "--image_or_video_path", type=str, help="The path of the image to be used as the background of the video.")
    add_argument_overridable(parser, "--lora_path", type=str, default=None, help="The path of the LoRA weights to be used")
    add_argument_overridable(parser, "--lora_rank", type=int, default=256, help="The rank of the LoRA weights")
    add_argument_overridable(parser, "--guidance_scale", type=float, default=1.0, help="The scale for classifier-free guidance")
    add_argument_overridable(parser, "--num_inference_steps", type=int, default=4, help="Inference steps")
    add_argument_overridable(parser, "--width", type=int, default=720, help="Number of steps for the inference process")
    add_argument_overridable(parser, "--height", type=int, default=480, help="Number of steps for the inference process")
    add_argument_overridable(parser, "--fps", type=int, default=16, help="Number of steps for the inference process")
    add_argument_overridable(parser, "--num_videos_per_prompt", type=int, default=1, help="Number of videos to generate per prompt")
    add_argument_overridable(parser, "--dtype", type=str, default="float16", help="The data type for computation")
    add_argument_overridable(parser, "--seed", type=int, default=42, help="The seed for reproducibility")
    # swin arguments
    add_argument_overridable(parser, "--num_noise_groups", type=int, default=4, help="Number of noise groups")
    add_argument_overridable(parser, "--num_sample_groups", type=int, default=8, help="Number of sampled videos groups")
    add_argument_overridable(parser, "--init_video_clip_frame", type=int, default=17, help="Frame number of init_video to be clipped, should be 4n+1")
    # lcm arguments
    add_argument_overridable(parser, "--original_inference_steps", type=int, default=40, help="Number of DDIM steps for training consistency model")
    add_argument_overridable(parser, "--lcm_multiplier", type=int, default=1, help="Number of lcm multiplier")
    # parallel arguments
    add_argument_overridable(parser, "--split_text_embed_in_sp", type=str, default="true", choices=["true", "false", "auto"], help="Whether to split text embed `encoder_hidden_states` for sequence parallel.")
    add_argument_overridable(parser, "--parallel_decoding_idx", type=int, default=-1, choices=[-1, 0, 1, 2, 3], help="Upblock index in VAE.decoder to enable parallel decoding. -1 means disabling parallel decoding.")
    add_argument_overridable(parser, "--wait_vae_seconds", type=float, default=0.0, help="Time that DiT wait for VAE.")
    args = parser.parse_args()
    print(f"[{FILE_NAME}.main] args parsed: {args}")

    engine_args = xFuserArgs.from_cli_args(args)
    engine_config, input_config = engine_args.create_config()
    local_rank = get_world_group().local_rank

    assert engine_args.pipefusion_parallel_degree == 1, "This script does not support PipeFusion."
    assert engine_args.use_parallel_vae is False, "parallel VAE not implemented for CogVideo"
    assert engine_config.runtime_config.use_torch_compile is False, "`use_torch_compile` is not supported yet."

    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    assert not args.enable_tiling and not args.enable_slicing, "Tiling and slicing are not supported yet."
    split_text_embed_in_sp = {
        "true": True,
        "false": False,
        "auto": None,
    }[args.split_text_embed_in_sp]
    parameter_peak_memory = torch.cuda.max_memory_allocated(device=f"cuda:{local_rank}")
    pipe = init_pipeline(
        engine_config=engine_config,
        input_config=input_config,
        model_path=args.model_path,
        local_rank=local_rank,
        lora_path=args.lora_path,
        lora_rank=args.lora_rank,
        dtype=dtype,
        enable_sequential_cpu_offload=args.enable_sequential_cpu_offload,
        enable_model_cpu_offload=args.enable_model_cpu_offload,
        enable_tiling=args.enable_tiling,
        enable_slicing=args.enable_slicing,
        parallel_decoding_idx=args.parallel_decoding_idx,
        split_text_embed_in_sp=split_text_embed_in_sp,
    )
    
    # args.prompt = "On a lush green meadow, a white car is driving. From an overhead panoramic shot, this car is adorned with blue and red stripes on its body, and it has a black spoiler at the rear. The camera follows the car as it moves through a field of golden wheat, surrounded by green grass and trees. In the distance, a river and some hills can be seen, with a cloudless blue sky above."
    gen_config = GenerationConfig(
        prompt=args.prompt,
        num_videos_per_prompt=args.num_videos_per_prompt,
        num_inference_steps=args.num_inference_steps,
        height=args.height,
        width=args.width,
        use_dynamic_cfg=False,  # This is used for DPM scheduler, for DDIM scheduler, it should be False
        guidance_scale=args.guidance_scale,
        generator_seed=args.seed,
        num_noise_groups=args.num_noise_groups,
        num_sample_groups=args.num_sample_groups,
        lcm_multiplier=args.lcm_multiplier,
        wait_vae_seconds=args.wait_vae_seconds,
        num_frames=args.init_video_clip_frame,
        fps=args.fps,
        video_cache_dir=args.video_cache_dir,
        default_video_path=args.image_or_video_path,
    )
    gen_config.kwargs = {"original_inference_steps": args.original_inference_steps}
    gen_config = pipe.prepare_generation_context(gen_config)
    
    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()
    
    pipe.start_interactive_loop(gen_config)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    peak_memory = torch.cuda.max_memory_allocated(device=f"cuda:{local_rank}")

    if get_world_group().rank == get_world_group().world_size - 1:
        print(f"epoch time: {elapsed_time:.2f} sec, parameter memory: {parameter_peak_memory/1e9:.2f} GB, memory: {peak_memory/1e9} GB")
    get_runtime_state().destory_distributed_env()


if __name__ == "__main__":
    main()