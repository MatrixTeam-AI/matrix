import argparse
from types import SimpleNamespace
from typing import Literal
import numpy as np
import os
import sys
sys.path.insert(0, '/'.join(os.path.realpath(__file__).split('/')[:-2]))
import hashlib
import torch
from stage4.cogvideox.pipelines import CogVideoXPipeline
from stage4.cogvideox.transformer import CogVideoXTransformer3DModel

from diffusers.utils import export_to_video, load_image, load_video


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

def generate_video(
    prompt: str,
    model_path: str,
    lora_path: str = None,
    lora_rank: int = 128,
    num_frames: int = 81,
    width: int = 720,
    height: int = 480,
    output_path: str = "./output.mp4",
    num_inference_steps: int = 50,
    guidance_scale: float = 6.0,
    num_videos_per_prompt: int = 1,
    dtype: torch.dtype = torch.bfloat16,
    seed: int = 42,
    fps: int = 8,
    gpu_id: int = 0,
    control_signal: str = None,
    control_seed: int = 42,
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
    transformer = CogVideoXTransformer3DModel.from_pretrained(
        os.path.join(model_path, "transformer"),
        torch_dtype=dtype,
        low_cpu_mem_usage=False
    )
    pipe = CogVideoXPipeline.from_pretrained(model_path, transformer=transformer, torch_dtype=dtype)

    # If you're using with lora, add this code
    if lora_path:
        pipe.load_lora_weights(lora_path, weight_name="pytorch_lora_weights.safetensors")
        pipe.fuse_lora(components=["transformer"],
            # lora_scale=1 / lora_rank  # It seems that there are some issues here, removed.
            )

    pipe.to(gpu_id)
    # pipe.enable_sequential_cpu_offload()
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    # 4. Generate the video frames based on the prompt.
    # `num_frames` is the Number of frames to generate.
    assert (num_frames - 1) % 4 == 0
    assert control_signal is not None
    # if control_signal is None:
    #     control_signal = generate_random_control_signal(int((num_frames - 1) / 4 + 1), seed=control_seed)
    print(f"Control signal: {control_signal}")

    video_generate = pipe(
        prompt=prompt,
        num_videos_per_prompt=num_videos_per_prompt,
        num_inference_steps=num_inference_steps,
        height=height,
        width=width,
        num_frames=num_frames,
        use_dynamic_cfg=True,
        guidance_scale=guidance_scale,
        generator=torch.Generator().manual_seed(seed),
        control_signal=control_signal
    ).frames[0]
    export_to_video(video_generate, output_path, fps=fps)

def fetch_base_video_path_by_prompt(prompt, cache_dir):
    prompt_hash_str = hashlib.sha256(prompt.encode()).hexdigest()
    print(f"Prompt hash: {prompt_hash_str}")
    video_path = os.path.join(cache_dir, f"output_{prompt_hash_str}.mp4")
    if os.path.exists(video_path):
        print(f"Base video already exists: {video_path}")
        return video_path, True
    else:
        return video_path, False
    
def generate_base_video(prompt, model_path, cache_dir):
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    video_path, exist = fetch_base_video_path_by_prompt(prompt, cache_dir)
    # check if there is a existing video with the same prompt
    if exist:
        print(f"Base video already exists at {video_path}")
    else:
        base_gen_config = SimpleNamespace(
            prompt=prompt,
            model_path=model_path,
            lora_path=None,  # e.g. "/path/to/lora.safetensors"
            lora_rank=256,
            output_path=video_path,
            guidance_scale=6.0,
            num_inference_steps=50,
            num_frames=65,
            width=720,
            height=480,
            fps=16,
            num_videos_per_prompt=1,
            dtype="bfloat16",        # "float16" for fp16
            seed=42,
            gpu_id=0,
            control_signal="D," * 16 + "D",  # 17 frame‑wise control tokens
            control_seed=42,
        )
        dtype = torch.float16 if base_gen_config.dtype == "float16" else torch.bfloat16
        print("output_path: ", base_gen_config.output_path)
        generate_video(
            prompt=base_gen_config.prompt,
            model_path=base_gen_config.model_path,
            lora_path=base_gen_config.lora_path,
            lora_rank=base_gen_config.lora_rank,
            output_path=base_gen_config.output_path,
            num_frames=base_gen_config.num_frames,
            width=base_gen_config.width,
            height=base_gen_config.height,
            num_inference_steps=base_gen_config.num_inference_steps,
            guidance_scale=base_gen_config.guidance_scale,
            num_videos_per_prompt=base_gen_config.num_videos_per_prompt,
            dtype=dtype,
            seed=base_gen_config.seed,
            fps=base_gen_config.fps,
            gpu_id=base_gen_config.gpu_id,
            control_signal=base_gen_config.control_signal,
            control_seed=base_gen_config.control_seed,
        )
        return video_path

if __name__ == "__main__":
    prompt_1 = "In a barren desert, a white SUV is driving. From an overhead panoramic shot, the vehicle has blue and red stripe decorations on its body, and there is a black spoiler at the rear. It is traversing through sand dunes and shrubs, kicking up a cloud of dust. In the distance, undulating mountains can be seen, with a sky of deep blue and a few white clouds floating by."
    prompt_2 = "On a lush green meadow, a white car is driving. From an overhead panoramic shot, this car is adorned with blue and red stripes on its body, and it has a black spoiler at the rear. The camera follows the car as it moves through a field of golden wheat, surrounded by green grass and trees. In the distance, a river and some hills can be seen, with a cloudless blue sky above."
    prompt_3 = "The video shows a white car driving on a country road on a sunny day. The car comes from the back of the scene, moving forward along the road, with open fields and distant hills surrounding it. As the car moves, the vegetation on both sides of the road and distant buildings can be seen. The entire video records the car's journey through the natural environment using a follow-shot technique."
    example_prompts = [prompt_1, prompt_2, prompt_3]
    
    for prompt in example_prompts:
        repo_root_dir = '/'.join(os.path.realpath(__file__).split('/')[:-2])
        cache_dir = os.path.join(repo_root_dir, "base_video_cache")
        model_path = os.path.join(repo_root_dir, "models/stage2")
        video_path = generate_base_video(prompt, model_path, cache_dir)