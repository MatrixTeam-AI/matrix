from dataclasses import dataclass, field, fields
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch

@dataclass
class DecodeConfig:
    output_type: str
    return_dict: bool
    
@dataclass
class GenerationStepConfig:
    prompt_embeds: torch.Tensor
    negative_prompt_embeds: torch.Tensor
    timesteps_grouped: torch.Tensor
    window_size: int
    inner_steps: int
    batch_size: int
    num_noise_groups: int
    num_warmup_steps: int
    with_frame_cond: bool
    lcm_multiplier: int
    do_classifier_free_guidance: bool
    image_rotary_emb: Optional[Tuple[torch.Tensor]] = None
    attention_kwargs: Optional[Dict] = None
    extra_step_kwargs: Optional[Dict] = None
    
@dataclass
class GenerationConfig:
    height: int
    width: int
    use_dynamic_cfg: bool
    num_videos_per_prompt: int
    num_inference_steps: int
    num_noise_groups: int
    num_frames: int  # TODO: Clarify if this is the number of frames for init video or num of latents in the window
    fps: int
    video_cache_dir: str
    default_video_path: str
    do_classifier_free_guidance: Optional[bool] = True
    num_sample_groups: Optional[int] = None
    with_frame_cond: bool = True
    eta: float = 0
    guidance_scale: float = 1.0
    output_type: str = "pil"
    return_dict: bool = True
    max_sequence_length: int = 226
    callback_on_step_end_tensor_inputs: List[str] = field(default_factory=lambda: ["latents"])
    prompt: Optional[str] = None
    init_video: Optional[torch.Tensor] = None
    negative_prompt: Optional[str] = None
    timesteps: Optional[torch.Tensor] = None
    generator: Optional[torch.Generator] = None
    generator_seed: Optional[int] = None
    latents: Optional[torch.Tensor] = None
    prompt_embeds: Optional[torch.Tensor] = None
    negative_prompt_embeds: Optional[torch.Tensor] = None
    attention_kwargs: Optional[dict] = None
    control_signal: Optional[List[str]] = None
    lcm_multiplier: Optional[int] = 1
    timesteps_grouped: Optional[torch.Tensor] = None
    batch_size: Optional[int] = None
    wait_vae_seconds: Optional[float] = 0.0
    kwargs: Dict = field(default_factory=dict)  # This field will catch all extra arguments
    gen_step_config: Optional[GenerationStepConfig] = None
    decode_config: Optional[DecodeConfig] = None
