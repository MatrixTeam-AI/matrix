# Copyright 2024 The CogVideoX team, Tsinghua University & ZhipuAI and The HuggingFace Team.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
import math
import random
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import sys
import os

import numpy as np
import torch
from transformers import T5EncoderModel, T5Tokenizer

from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.models.embeddings import get_3d_rotary_pos_embed
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.utils import logging, replace_example_docstring
from diffusers.utils.torch_utils import randn_tensor
from diffusers.video_processor import VideoProcessor

sys.path.insert(0, '/'.join(os.path.realpath(__file__).split('/')[:-3]))
from stage4.cogvideox.pipelines.pipeline_output import CogVideoXPipelineOutput
from stage4.cogvideox.loader import CogVideoXLoraLoaderMixin
from stage4.cogvideox.autoencoder import AutoencoderKLCogVideoX
from stage4.cogvideox.transformer import CogVideoXTransformer3DModel
from stage4.cogvideox.scheduler import (
    LCMSwinScheduler,
    CogVideoXDPMScheduler,
    CogVideoXSwinDPMScheduler,
    expand_timesteps_with_group,
)
from stage4.cogvideox.control_adapter import CONTROL_SIGNAL_TO_PROMPT

from transformers import AutoTokenizer, CLIPModel

from contextlib import nullcontext

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

from dataclasses import dataclass, field, fields
# ============================
import gc
import time
from typing import Union
import ray
from xfuser.core.distributed import (
    get_world_group,
)
from xfuser.core.distributed.group_coordinator import GroupCoordinator
import torch.cuda.nvtx as nvtx
from journee.utils.ray_pipeline_utils import timer, add_timestamp, get_data_and_timestamps, get_passed_times, add_timestamp_to_each_item, QueueManager, SharedVar, SharedReadOnlyVar
from journee.stage4_ray.config import GenerationConfig, GenerationStepConfig, DecodeConfig
from journee.generate_base_video import generate_base_video, fetch_base_video_path_by_prompt
import decord
import PIL.Image
### set print to a dummy function ###
# def print(*args, **kwargs):
#     pass
# ============================
EXAMPLE_DOC_STRING = ""

def load_init_video_clip(
    image_or_video_path: str,
    init_video_clip_frame: int,
    fps: int,
):
    video_reader = decord.VideoReader(image_or_video_path)
    video_num_frames = len(video_reader)
    video_fps = video_reader.get_avg_fps()
    sampling_interval = video_fps/fps
    frame_indices = np.round(np.arange(0, video_num_frames, sampling_interval)).astype(int).tolist()
    frame_indices = frame_indices[-init_video_clip_frame:]
    video = video_reader.get_batch(frame_indices).asnumpy()
    video = [PIL.Image.fromarray(frame) for frame in video]
    init_num_video_frames = len(video)
    assert init_num_video_frames == init_video_clip_frame, f"init_video_clip_frame should be {init_num_video_frames}, but got {init_video_clip_frame}"
    return video

def randn_like(tensor, generator=None):
    return randn_tensor(
        tensor.shape,
        generator=generator, dtype=tensor.dtype, device=tensor.device
    )

# Similar to diffusers.pipelines.hunyuandit.pipeline_hunyuandit.get_resize_crop_region_for_grid
def get_resize_crop_region_for_grid(src, tgt_width, tgt_height):
    tw = tgt_width
    th = tgt_height
    h, w = src
    r = h / w
    if r > (th / tw):
        resize_height = th
        resize_width = int(round(th / h * w))
    else:
        resize_width = tw
        resize_height = int(round(tw / w * h))

    crop_top = int(round((th - resize_height) / 2.0))
    crop_left = int(round((tw - resize_width) / 2.0))

    return (crop_top, crop_left), (crop_top + resize_height, crop_left + resize_width)


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    r"""
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    

    return timesteps, num_inference_steps


class CogVideoXPipeline(DiffusionPipeline, CogVideoXLoraLoaderMixin):
    r"""
    Pipeline for text-to-video generation using CogVideoX.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode videos to and from latent representations.
        text_encoder ([`T5EncoderModel`]):
            Frozen text-encoder. CogVideoX uses
            [T5](https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5EncoderModel); specifically the
            [t5-v1_1-xxl](https://huggingface.co/PixArt-alpha/PixArt-alpha/tree/main/t5-v1_1-xxl) variant.
        tokenizer (`T5Tokenizer`):
            Tokenizer of class
            [T5Tokenizer](https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5Tokenizer).
        transformer ([`CogVideoXTransformer3DModel`]):
            A text conditioned `CogVideoXTransformer3DModel` to denoise the encoded video latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `transformer` to denoise the encoded video latents.
    """

    _optional_components = []
    model_cpu_offload_seq = "text_encoder->transformer->vae"

    _callback_tensor_inputs = [
        "latents",
        "prompt_embeds",
        "negative_prompt_embeds",
    ]

    def __init__(
        self,
        tokenizer: T5Tokenizer,
        text_encoder: T5EncoderModel,
        vae: AutoencoderKLCogVideoX,
        transformer: CogVideoXTransformer3DModel,
        scheduler: CogVideoXDPMScheduler,
    ):
        super().__init__()

        self.register_modules(
            tokenizer=tokenizer, text_encoder=text_encoder, vae=vae, transformer=transformer, scheduler=scheduler
        )
        self.vae_scale_factor_spatial = (
            2 ** (len(self.vae.config.block_out_channels) - 1) if hasattr(self, "vae") and self.vae is not None else 8
        )
        self.vae_scale_factor_temporal = (
            self.vae.config.temporal_compression_ratio if hasattr(self, "vae") and self.vae is not None else 4
        )
        self.vae_scaling_factor_image = (
            self.vae.config.scaling_factor if hasattr(self, "vae") and self.vae is not None else 0.7
        )

        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)

    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        num_videos_per_prompt: int = 1,
        max_sequence_length: int = 226,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        device = device or self._execution_device
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer.batch_decode(untruncated_ids[:, max_sequence_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because `max_sequence_length` is set to "
                f" {max_sequence_length} tokens: {removed_text}"
            )

        prompt_embeds = self.text_encoder(text_input_ids.to(device))[0]
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

        return prompt_embeds

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        do_classifier_free_guidance: bool = True,
        num_videos_per_prompt: int = 1,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        max_sequence_length: int = 226,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            do_classifier_free_guidance (`bool`, *optional*, defaults to `True`):
                Whether to use classifier free guidance or not.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                Number of videos that should be generated per prompt. torch device to place the resulting embeddings on
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            device: (`torch.device`, *optional*):
                torch device
            dtype: (`torch.dtype`, *optional*):
                torch dtype
        """
        device = device or self._execution_device

        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_embeds = self._get_t5_prompt_embeds(
                prompt=prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt

            if prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )

            negative_prompt_embeds = self._get_t5_prompt_embeds(
                prompt=negative_prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        return prompt_embeds, negative_prompt_embeds

    def prepare_latents(
        self, batch_size, num_channels_latents, num_frames, height, width, dtype, device, generator, latents=None
    ):
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        shape = (
            batch_size,
            (num_frames - 1) // self.vae_scale_factor_temporal + 1,
            num_channels_latents,
            height // self.vae_scale_factor_spatial,
            width // self.vae_scale_factor_spatial,
        )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        latents = latents.permute(0, 2, 1, 3, 4)  # [batch_size, num_channels, num_frames, height, width]
        latents = 1 / self.vae_scaling_factor_image * latents

        frames = self.vae.decode(latents).sample
        return frames

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs
    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    # Copied from diffusers.pipelines.latte.pipeline_latte.LattePipeline.check_inputs
    def check_inputs(
        self,
        prompt,
        height,
        width,
        negative_prompt,
        callback_on_step_end_tensor_inputs,
        prompt_embeds=None,
        negative_prompt_embeds=None,
    ):
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )
        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

    def fuse_qkv_projections(self) -> None:
        r"""Enables fused QKV projections."""
        self.fusing_transformer = True
        self.transformer.fuse_qkv_projections()

    def unfuse_qkv_projections(self) -> None:
        r"""Disable QKV projection fusion if enabled."""
        if not self.fusing_transformer:
            logger.warning("The Transformer was not initially fused for QKV projections. Doing nothing.")
        else:
            self.transformer.unfuse_qkv_projections()
            self.fusing_transformer = False

    def _prepare_rotary_positional_embeddings(
        self,
        height: int,
        width: int,
        num_frames: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        grid_height = height // (self.vae_scale_factor_spatial * self.transformer.config.patch_size)
        grid_width = width // (self.vae_scale_factor_spatial * self.transformer.config.patch_size)
        base_size_width = 720 // (self.vae_scale_factor_spatial * self.transformer.config.patch_size)
        base_size_height = 480 // (self.vae_scale_factor_spatial * self.transformer.config.patch_size)

        grid_crops_coords = get_resize_crop_region_for_grid(
            (grid_height, grid_width), base_size_width, base_size_height
        )
        freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
            embed_dim=self.transformer.config.attention_head_dim,
            crops_coords=grid_crops_coords,
            grid_size=(grid_height, grid_width),
            temporal_size=num_frames,
        )

        freqs_cos = freqs_cos.to(device=device)
        freqs_sin = freqs_sin.to(device=device)
        return freqs_cos, freqs_sin

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def attention_kwargs(self):
        return self._attention_kwargs

    @property
    def interrupt(self):
        return self._interrupt
    
    def get_control_from_signal(self, control_signal):
        if not hasattr(self, "control_embeddings"):
            clip_tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-large-patch14")
            clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
            clip_model.requires_grad_(False)
            clip_model.to(self._execution_device, dtype=self.transformer.dtype)

            control_prompts = list(CONTROL_SIGNAL_TO_PROMPT.values())
            control_prompt_ids = clip_tokenizer(control_prompts, padding=True, return_tensors="pt")
            control_prompt_ids.to(self._execution_device)
            self.control_embeddings = clip_model.get_text_features(**control_prompt_ids)
            null_control_prompt_ids = clip_tokenizer([""], padding=True, return_tensors="pt")
            null_control_prompt_ids.to(self._execution_device)
            self.null_control_embedding = clip_model.get_text_features(**null_control_prompt_ids)
            self.control_signal_to_idx = {k: i for i, k in enumerate(CONTROL_SIGNAL_TO_PROMPT.keys())}

        control_indices = [self.control_signal_to_idx[c] for c in control_signal.split(",")]
        control = self.control_embeddings[control_indices]
        return control

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        height: int = 480,
        width: int = 720,
        num_frames: int = 49,
        num_inference_steps: int = 50,
        timesteps: Optional[List[int]] = None,
        guidance_scale: float = 6,
        use_dynamic_cfg: bool = False,
        num_videos_per_prompt: int = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: str = "pil",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 226,
        control_signal: List[str] = None
    ) -> Union[CogVideoXPipelineOutput, Tuple]:
        """
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            height (`int`, *optional*, defaults to self.transformer.config.sample_height * self.vae_scale_factor_spatial):
                The height in pixels of the generated image. This is set to 480 by default for the best results.
            width (`int`, *optional*, defaults to self.transformer.config.sample_height * self.vae_scale_factor_spatial):
                The width in pixels of the generated image. This is set to 720 by default for the best results.
            num_frames (`int`, defaults to `48`):
                Number of frames to generate. Must be divisible by self.vae_scale_factor_temporal. Generated video will
                contain 1 extra frame because CogVideoX is conditioned with (num_seconds * fps + 1) frames where
                num_seconds is 6 and fps is 8. However, since videos can be saved at any fps, the only condition that
                needs to be satisfied is that of divisibility mentioned above.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
                in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
                passed will be used. Must be in descending order.
            guidance_scale (`float`, *optional*, defaults to 7.0):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                The number of videos to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion_xl.StableDiffusionXLPipelineOutput`] instead
                of a plain tuple.
            attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            max_sequence_length (`int`, defaults to `226`):
                Maximum sequence length in encoded prompt. Must be consistent with
                `self.transformer.config.max_text_seq_length` otherwise may lead to poor results.

        Examples:

        Returns:
            [`~pipelines.cogvideo.pipeline_cogvideox.CogVideoXPipelineOutput`] or `tuple`:
            [`~pipelines.cogvideo.pipeline_cogvideox.CogVideoXPipelineOutput`] if `return_dict` is True, otherwise a
            `tuple`. When returning a tuple, the first element is a list with the generated images.
        """

        # if num_frames > 49:
        #     raise ValueError(
        #         "The number of frames must be less than 49 for now due to static positional embeddings. This will be updated in the future to remove this limitation."
        #     )

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        num_videos_per_prompt = 1

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            height,
            width,
            negative_prompt,
            callback_on_step_end_tensor_inputs,
            prompt_embeds,
            negative_prompt_embeds,
        )
        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._interrupt = False

        # 2. Default call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            negative_prompt,
            do_classifier_free_guidance,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            max_sequence_length=max_sequence_length,
            device=device,
        )
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        
        # Control signal
        control_emb = self.get_control_from_signal(control_signal)
        control_emb = control_emb.unsqueeze(0).to(prompt_embeds.dtype).contiguous()

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps)
        self._num_timesteps = len(timesteps)

        # 5. Prepare latents.
        latent_channels = self.transformer.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            latent_channels,
            num_frames,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Create rotary embeds if required
        image_rotary_emb = (
            self._prepare_rotary_positional_embeddings(height, width, latents.size(1), device)
            if self.transformer.config.use_rotary_positional_embeddings
            else None
        )

        # 8. Denoising loop
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            # for DPM-solver++
            old_pred_original_sample = None
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latent_model_input.shape[0])

                # predict noise model_output
                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    encoder_hidden_states=prompt_embeds,
                    timestep=timestep,
                    image_rotary_emb=image_rotary_emb,
                    attention_kwargs=attention_kwargs,
                    return_dict=False,
                    control_emb=control_emb
                )[0]
                noise_pred = noise_pred.float()

                # perform guidance
                if use_dynamic_cfg:
                    self._guidance_scale = 1 + guidance_scale * (
                        (1 - math.cos(math.pi * ((num_inference_steps - t.item()) / num_inference_steps) ** 5.0)) / 2
                    )
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                if not isinstance(self.scheduler, CogVideoXDPMScheduler):
                    latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
                else:
                    latents, old_pred_original_sample = self.scheduler.step(
                        noise_pred,
                        old_pred_original_sample,
                        t,
                        timesteps[i - 1] if i > 0 else None,
                        latents,
                        **extra_step_kwargs,
                        return_dict=False,
                    )
                latents = latents.to(prompt_embeds.dtype)

                # call the callback, if provided
                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        if not output_type == "latent":
            video = self.decode_latents(latents)
            video = self.video_processor.postprocess_video(video=video, output_type=output_type)
        else:
            video = latents

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (video,)

        return CogVideoXPipelineOutput(frames=video)


from PIL import Image
from typing import Union


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.retrieve_latents
def retrieve_latents(
    encoder_output: torch.Tensor,
    generator: Optional[torch.Generator] = None,
    sample_mode: str = "sample",
):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")


class CogVideoXInteractiveStreamingPipeline(CogVideoXPipeline):
    def __init__(
        self,
        tokenizer: T5Tokenizer,
        text_encoder: T5EncoderModel,
        vae: AutoencoderKLCogVideoX,
        transformer: CogVideoXTransformer3DModel,
        scheduler: Union[CogVideoXDPMScheduler, CogVideoXSwinDPMScheduler],
    ):
        super().__init__(
            tokenizer,
            text_encoder,
            vae,
            transformer,
            scheduler,
        )
        
        # ===== for interactive streaming =====
        # The return of get_world_group() is a custom GroupCoordinator object
        self.dit_process_group = get_world_group()  # TODO: this should be replaced with `get_dit_group` in the ray environment
        self.rank = self.dit_process_group.rank
        
        self.set_up_control_embeddings()
        self.init_ray_sender()
        self.init_action = 'D'
        self.model_input_actions = []
        self.control_signal_to_idx = {k: i for i, k in enumerate(CONTROL_SIGNAL_TO_PROMPT.keys())}
        self.idx_to_control_signal = {v : k for k, v in self.control_signal_to_idx.items()}
        self.action_cache = []
        # ========================================
    
    def init_ray_sender(self):  # for interactive streaming
        if self.rank == 0:  # assert the total worker rank order is [dit_workers=M, vae_workers=N, postprocessor_worker=1], so the first worker is the dit_worker
            ray.init(address='auto')  # connect to ray cluster
            self.queue_manager = ray.get_actor("dit2vae_queue", namespace='matrix')
            self.action_manager = ray.get_actor("action_queue", namespace='matrix') 
            self.dit_step_var = ray.get_actor("dit_step_var", namespace='matrix')
            self.vae_step_var = ray.get_actor("vae_step_var", namespace='matrix')
            
            self.current_state_var = ray.get_actor(namespace='matrix', name="current_state")
            self.current_prompt_var = ray.get_actor(namespace='matrix', name="current_prompt")
            
            self.vae2post_queue = ray.get_actor(namespace='matrix', name="vae2post_queue")  # VAE --> Postprocessing
            self.post2front_queue = ray.get_actor(namespace='matrix', name="post2front_queue")  # Postprocessing --> front end
        
    def send_latents_to_queue(self, latents, batch_timestamps):   # for interactive streaming
        if self.rank == 0:  # TODO: Try not to use ray.get to avoid blocking to save time?
            assert hasattr(self, "queue_manager")
            latents = add_timestamp(latents, label='DiT-out-latent', timestamps=batch_timestamps)
            ray.get(self.queue_manager.put.remote(latents))

    def fetch_all_actions(self, window_size):  # for interactive streaming
        # get all actions from front-end, and keep the latest ones
        if self.rank == 0:
            assert len(self.action_cache) == 0
            while len(self.action_cache) < window_size:
                actions = ray.get(self.action_manager.get_all.remote())  # this may return empty list []
                actions = add_timestamp_to_each_item(actions, label='DiT-in-control')
                if len(actions) == 0:
                    action = ray.get(self.action_manager.get.remote())  # this is blocking
                    action = add_timestamp(action, label='DiT-in-control')
                    actions.append(action)
                self.action_cache.extend(actions)
            self.action_cache = self.action_cache[-window_size:]
        torch.distributed.barrier()
    
    def _get_action(self):  # for interactive streaming
        action = self.action_cache.pop(0)
        action, timestamps = get_data_and_timestamps(action)
        if timestamps is not None:
            passed_times = get_passed_times(timestamps)
            if passed_times:
                self.print(f"[CogVideoXInteractiveStreamingPipeline._get_action] passed_times:\n{passed_times}")
        return action, timestamps
    
    def get_current_action(self):  # for interactive streaming
        # Get the current action input from the action shared variable and broadcast it to all other dit workers
        # return type: str, e.g. "D"
        device = torch.device(f'cuda:{self.dit_process_group.local_rank}')
        # print("Rank: ", self.rank, "Device: ", device)
        # print("current dit group: ", self.dit_process_group.ranks)
        timestamps = None
        if self.rank == 0:
            assert hasattr(self, "action_manager")
            # print("[RANK 0] Send action to all other dit workers")
            with timer(label=f"[RANK {self.rank}]: `self._get_action`"):
                current_action, timestamps = self._get_action()
            cur_action_id = self.control_signal_to_idx.get(current_action, 0)
            # Broadcast the action id to all other dit workers
            with timer(label=f"[RANK {self.rank}]: Broadcasting current action"):
                cur_action_id_tensor = torch.tensor([cur_action_id], dtype=torch.int, device=device)
                assert isinstance(self.dit_process_group.device_group, torch.distributed.ProcessGroup), f"Invalid process group type: {type(self.dit_process_group.device_group)}"
                torch.distributed.broadcast(cur_action_id_tensor, src=0, group=self.dit_process_group.device_group)
        else:
            # print(f"[RANK {self.rank}] Receive action from rank 0")
            # Receive the action id from the first dit worker
            cur_action_id_tensor = torch.zeros(1, dtype=torch.int, device=device)
            torch.distributed.broadcast(cur_action_id_tensor, src=0, group=self.dit_process_group.device_group)
            # Convert the action id to action string
            current_action = self.idx_to_control_signal[cur_action_id_tensor.item()]
        torch.distributed.barrier()
        self.print(f"current_action: {current_action}")
        assert current_action in ["D", "DR", "DL", "N", "B"], f"Invalid action input: {current_action}"
        return current_action, timestamps

    def print(self, *args, **kwargs):  # for interactive streaming
        if self.rank == 0:
            print(*args, **kwargs)
    
    def get_control_from_signal_interactive(self, gen_config:GenerationConfig):  # for interactive streaming
        num_latent_frames, window_size = gen_config.num_frames, gen_config.gen_step_config.window_size
        # TODO: Handle this in multi-gpu setting
        assert hasattr(self, "control_embeddings")
        if len(self.model_input_actions) == 0:  # initialize the action window for the first time
            self.model_input_actions = [self.init_action] * num_latent_frames
            self.model_input_action_timestamps = None
            if self.rank == 0:
                self.model_input_action_timestamps = [[] for _ in range(num_latent_frames)]
        else:
            action_window = []
            timestamps_window = []
            with timer(label=f"[RANK {self.rank}]: `self.fetch_all_actions`"):
                self.fetch_all_actions(window_size)
            for _ in range(window_size):
                # this will block the process if the initial value is None, continue until frontend updates the value from keyboard
                action, timestamps = self.get_current_action()
                action_window.append(action)
                timestamps_window.append(timestamps)
            self.model_input_actions.extend(action_window)
            self.model_input_actions = self.model_input_actions[-num_latent_frames:]
            if self.model_input_action_timestamps is not None:
                self.model_input_action_timestamps.extend(timestamps_window)
                self.model_input_action_timestamps = self.model_input_action_timestamps[-num_latent_frames:]
        self.print(f"Current actions: {self.model_input_actions}")
        control_indices = [self.control_signal_to_idx[action] for action in self.model_input_actions] 
        control_emb = self.control_embeddings[control_indices]
        control_emb = control_emb.unsqueeze(0).to(gen_config.gen_step_config.prompt_embeds.dtype).contiguous()
        if gen_config.do_classifier_free_guidance:
            control_emb = torch.cat([control_emb, control_emb], dim=0)
        return control_emb

    def pop_timestamps(self, window_size, with_frame_cond=True):  # for interactive streaming
        if self.model_input_action_timestamps is None:
            return None
        if with_frame_cond:
            return self.model_input_action_timestamps[1 : 1 + window_size]
        else:
            return self.model_input_action_timestamps[ : window_size]

    def wait(self, group_idx, sec):  # for interactive streaming
        # wait for the preparation of VAE
        torch.cuda.synchronize()
        # if group_idx == 1:
        #     vae_warmup_time = 30
        #     self.print(f"Waiting {vae_warmup_time} seconds for VAE warmup...")
        #     time.sleep(vae_warmup_time)
        if sec > 0:
            time.sleep(sec)
            
    def prepare_latents(
        self,
        init_video: Optional[torch.Tensor] = None,
        batch_size: int = 1,
        num_channels_latents: int = 16,
        num_frames: int = 49,
        height: int = 60,
        width: int = 90,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.Tensor] = None,
        timestep: Optional[torch.Tensor] = None,
    ):
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        shape = (
            batch_size,
            num_frames,
            num_channels_latents,
            height // self.vae_scale_factor_spatial,
            width // self.vae_scale_factor_spatial,
        )

        if latents is None:
            if isinstance(generator, list):
                if len(generator) != batch_size:
                    raise ValueError(
                        f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                        f" size of {batch_size}. Make sure the batch size matches the length of the generators."
                    )

                video_latents = [retrieve_latents(self.vae.encode(init_video[i].unsqueeze(0)), generator[i]) for i in range(batch_size)]
            else:
                video_latents = [retrieve_latents(self.vae.encode(vid.unsqueeze(0)), generator) for vid in init_video]

            video_latents = torch.cat(video_latents, dim=0).to(dtype).permute(0, 2, 1, 3, 4)  # [B, F, C, H, W]
            video_latents = self.vae_scaling_factor_image * video_latents
            if video_latents.size(1) < num_frames:
                padding_shape = (
                    batch_size,
                    num_frames - video_latents.size(1),
                    num_channels_latents,
                    height // self.vae_scale_factor_spatial,
                    width // self.vae_scale_factor_spatial,
                )
                latent_padding = torch.zeros(padding_shape, device=device, dtype=dtype)
                init_latents = torch.cat([video_latents, latent_padding], dim=1)
            else:
                init_latents = video_latents[:, :num_frames]

            # add grouped noise
            noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            latents = self.scheduler.add_noise(init_latents, noise, timestep.unsqueeze(0).repeat(batch_size, 1))
        else:
            # add grouped noise
            noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            latents = self.scheduler.add_noise(latents, noise, timestep.unsqueeze(0).repeat(batch_size, 1))
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents
    
    def set_up_control_embeddings(self):
        if not hasattr(self, "control_embeddings"):
            clip_tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-large-patch14")
            clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
            clip_model.requires_grad_(False)
            device = torch.device(f'cuda:{self.rank}')
            clip_model.to(device, dtype=self.transformer.dtype)

            control_prompts = list(CONTROL_SIGNAL_TO_PROMPT.values())
            control_prompt_ids = clip_tokenizer(control_prompts, padding=True, return_tensors="pt")
            control_prompt_ids.to(device)
            self.control_embeddings = clip_model.get_text_features(**control_prompt_ids)
            null_control_prompt_ids = clip_tokenizer([""], padding=True, return_tensors="pt")
            null_control_prompt_ids.to(device)
            self.null_control_embedding = clip_model.get_text_features(**null_control_prompt_ids)
            
            # delete clip model after getting all control embeddings
            del clip_model
            gc.collect()
            torch.cuda.empty_cache()
        
    def get_control_from_signal(self, control_signal, start=None, end=None):
        assert hasattr(self, "control_embeddings")

        control_signal_list = control_signal.split(",")
        if start is not None and end is not None:
            control_signal_list = control_signal_list[start: end]
        control_indices = [self.control_signal_to_idx[c] for c in control_signal_list]
        control = self.control_embeddings[control_indices]
        return control, control_indices
    
    def decode_latents(self, latents: torch.Tensor, sliced_decode=False, keep_cache=True) -> torch.Tensor:
        latents = latents.permute(0, 2, 1, 3, 4)  # [batch_size, num_channels, num_frames, height, width]
        latents = 1 / self.vae_scaling_factor_image * latents

        if sliced_decode:
            batch_size, num_channels, num_latent_frames, height, width = latents.size()
            assert batch_size == 1, "Only support one video to decode when using sliced decoding"
            decode_window = 128  # can be any length
            overlap_length = 4
            cur_start = 0
            all_frames = []
            while cur_start + decode_window <= num_latent_frames:
                cur_frames = self.vae.decode(latents[:, :, cur_start:cur_start+decode_window], keep_cache=keep_cache).sample
                cur_frames_cpu = cur_frames.cpu()
                del cur_frames
                all_frames.append(cur_frames_cpu)
                cur_start += (decode_window - overlap_length)
            if cur_start + overlap_length < num_latent_frames or cur_start == 0:
                cur_frames = self.vae.decode(latents[:, :, cur_start:], keep_cache=keep_cache).sample
                cur_frames_cpu = cur_frames.cpu()
                del cur_frames
                all_frames.append(cur_frames_cpu)
            frames = [all_frames[0]]
            frames.extend([item[:, :, overlap_length*4:] for item in all_frames[1:]])
            frames = torch.cat(frames, dim=2)
        else:
            frames = self.vae.decode(latents, keep_cache=keep_cache).sample
        
        return frames
    
    @torch.no_grad()
    def generate_step_interactive(self, latents, control_emb, group_idx, gen_config):
        gen_step_config = gen_config.gen_step_config
        (num_noise_groups, lcm_multiplier, num_warmup_steps, with_frame_cond, window_size, inner_steps, timesteps_grouped, 
         batch_size, do_classifier_free_guidance, prompt_embeds, negative_prompt_embeds, image_rotary_emb, 
         attention_kwargs, extra_step_kwargs) = (gen_step_config.num_noise_groups, gen_step_config.lcm_multiplier, gen_step_config.num_warmup_steps, gen_step_config.with_frame_cond, gen_step_config.window_size, \
                        gen_step_config.inner_steps, gen_step_config.timesteps_grouped, gen_step_config.batch_size, gen_step_config.do_classifier_free_guidance, gen_step_config.prompt_embeds, gen_step_config.negative_prompt_embeds, gen_step_config.image_rotary_emb, \
                        gen_step_config.attention_kwargs, gen_step_config.extra_step_kwargs)
        generator = extra_step_kwargs['generator']
        dit_start_time = time.time()
        nvtx.range_push(f"Decoding of {group_idx}th latent")
        with self.progress_bar(total=inner_steps) as progress_bar:  # inner_steps = num_inference_steps // num_noise_groups
            # for DPM-solver++
            old_pred_original_sample = None
            for i in range(inner_steps):
                if self.interrupt:
                    continue
                # [B, F', 1, 1, 1]
                timesteps = timesteps_grouped[i : i + 1].repeat(batch_size, 1)[..., None, None, None]

                if i > 0:
                    # [B, F', 1, 1, 1]
                    timesteps_back = timesteps_grouped[i - 1 : i].repeat(batch_size, 1)[..., None, None, None]
                else:
                    timesteps_back = None

                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                # latent_model_input = self.scheduler.scale_model_input(
                #     latent_model_input, t
                # )

                # predict noise model_output
                with timer(label=f"[RANK {self.rank}]: Only DiT with {latent_model_input.shape=}"): 
                    noise_pred = self.transformer(
                        hidden_states=latent_model_input,
                        encoder_hidden_states=prompt_embeds,
                        timestep=timesteps,
                        image_rotary_emb=image_rotary_emb,
                        attention_kwargs=attention_kwargs,
                        return_dict=False,
                        control_emb=control_emb,
                    )[0]
                    noise_pred = noise_pred.float()

                # perform guidance
                # issue with strange logic: https://github.com/huggingface/diffusers/issues/9641
                # if use_dynamic_cfg:
                #     self._guidance_scale = 1 + guidance_scale * (
                #         (1 - math.cos(math.pi * ((num_inference_steps - t.item()) / num_inference_steps) ** 5.0)) / 2
                #     )
                if do_classifier_free_guidance:
                    guidance_scale = guidance_scale * torch.ones(batch_size)
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents_cond = latents[:, :1]
                latents, old_pred_original_sample = self.scheduler.step(
                    noise_pred[:,1:],
                    timesteps[:,1:],
                    latents[:,1:],
                    **extra_step_kwargs,
                    return_dict=False,
                    num_lcm_phases=num_noise_groups * lcm_multiplier
                )
                latents = latents.to(prompt_embeds.dtype)
                latents = torch.cat([latents_cond, latents], dim=1)  # keep cond frame unchanged

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
        torch.cuda.synchronize()
        nvtx.range_pop()
        dit_end_time = time.time()
        self.print(f"Group_{group_idx} DiT time: {dit_end_time - dit_start_time}")
        if(self.rank == 0):
            self.dit_step_var.set.remote(new_value=group_idx)
        if with_frame_cond:
            latents_pop = latents[:, 1 : window_size + 1]
            latents_remain = latents[:, window_size + 1 :]
            latents_cond_new = latents_pop[:, -1].unsqueeze(1)
            latents_new = torch.cat(
                [latents_cond_new, latents_remain, randn_like(latents_pop, generator)],
                dim=1,
            )  # append noisy video token to the right of the video sequence
            latents = latents_new
        else:
            latents_pop = latents[:, :window_size]
            latents_remain = latents[:, window_size:]
            latents_new = torch.cat([latents_remain, randn_like(latents_pop, generator)], dim=1)
            latents = latents_new
        return latents, latents_pop
    
    def prepare_prompt_embeds(self, gen_config: GenerationConfig):
        device = self._execution_device
        prompt, negative_prompt, guidance_scale, do_classifier_free_guidance, num_videos_per_prompt, max_sequence_length, prompt_embeds, negative_prompt_embeds = (
            gen_config.prompt, gen_config.negative_prompt, gen_config.guidance_scale, gen_config.do_classifier_free_guidance, gen_config.num_videos_per_prompt, gen_config.max_sequence_length, gen_config.prompt_embeds, gen_config.negative_prompt_embeds
        )
        print("Prompt: ", prompt, negative_prompt, guidance_scale, do_classifier_free_guidance, num_videos_per_prompt, max_sequence_length, prompt_embeds, negative_prompt_embeds)
        do_classifier_free_guidance = do_classifier_free_guidance and (guidance_scale > 1.0)

        # 3. Encode input prompt
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            negative_prompt,
            do_classifier_free_guidance,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            max_sequence_length=max_sequence_length,
            device=device,
        )

        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        
        # update the prompt_embeds and negative_prompt_embeds in gen_config
        gen_config.gen_step_config.prompt_embeds = prompt_embeds
        gen_config.gen_step_config.negative_prompt_embeds = negative_prompt_embeds
        
        # return prompt_embeds, negative_prompt_embeds
        return gen_config
    
    def prepare_latents_from_video(self, gen_config: GenerationConfig, device=None, dtype=None):
        (latents, init_video, height, width, batch_size, num_videos_per_prompt, num_frames, timesteps_grouped, generator) = (
            gen_config.latents, gen_config.init_video, gen_config.height, gen_config.width, gen_config.batch_size, gen_config.num_videos_per_prompt, gen_config.num_frames, gen_config.timesteps_grouped, gen_config.generator)
        device = device or self._execution_device
        dtype = dtype or self.text_encoder.dtype
        if latents is None:
            init_video = self.video_processor.preprocess_video(init_video, height=height, width=width)
            init_video = init_video.to(device=device, dtype=dtype)
            latents = None
        else:
            init_video = None
            latents = latents.to(device, dtype=dtype)

        latent_channels = self.transformer.config.in_channels
        latents = self.prepare_latents(  # add noise to the latents (from video clip) 
            init_video,
            batch_size * num_videos_per_prompt,
            latent_channels,
            num_frames,
            height,
            width,
            dtype,
            device,
            generator,
            latents,
            timesteps_grouped[0],
        )
        print("Latent size:", latents.size())
        return latents
    
    def prepare_generation_context(
        self,
        gen_config: GenerationConfig
        )->Tuple[GenerationStepConfig, DecodeConfig]:
        
        init_video, prompt, negative_prompt, height, width, num_frames, num_inference_steps, timesteps, guidance_scale, use_dynamic_cfg, num_videos_per_prompt, eta, generator, latents, prompt_embeds, negative_prompt_embeds, output_type, return_dict, attention_kwargs, callback_on_step_end_tensor_inputs, max_sequence_length, control_signal, num_noise_groups, num_sample_groups, with_frame_cond, do_classifier_free_guidance, lcm_multiplier, kwargs = (
            gen_config.init_video, gen_config.prompt, gen_config.negative_prompt, gen_config.height, gen_config.width, gen_config.num_frames, gen_config.num_inference_steps, gen_config.timesteps, gen_config.guidance_scale, gen_config.use_dynamic_cfg, gen_config.num_videos_per_prompt, gen_config.eta, gen_config.generator, gen_config.latents, gen_config.prompt_embeds, gen_config.negative_prompt_embeds, gen_config.output_type, gen_config.return_dict, gen_config.attention_kwargs, gen_config.callback_on_step_end_tensor_inputs, gen_config.max_sequence_length, gen_config.control_signal, gen_config.num_noise_groups, gen_config.num_sample_groups, gen_config.with_frame_cond, gen_config.do_classifier_free_guidance, gen_config.lcm_multiplier, gen_config.kwargs
        )
        assert isinstance(self.scheduler, LCMSwinScheduler)
        assert with_frame_cond, "Currently `with_frame_cond` must be True."
        num_videos_per_prompt = 1
        outer_steps = num_sample_groups
        if with_frame_cond:
            num_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1 if latents is None else latents.size(1)    
            
            inner_steps = num_inference_steps // num_noise_groups
            window_size = (num_frames - 1) // num_noise_groups
        else:
            num_frames = num_frames // self.vae_scale_factor_temporal if latents is None else latents.size(1)
            inner_steps = num_inference_steps // num_noise_groups
            window_size = num_frames // num_noise_groups
        gen_config.num_frames = num_frames
        
        if with_frame_cond:    
            num_frames_nocond = num_frames - 1
        else:
            num_frames_nocond = num_frames
        
        assert (
            num_frames_nocond % num_noise_groups == 0
        ), f"num_frames (without conditional frame) should be divisible by num_noise_groups, but get: num_frames_nocond {num_frames_nocond} and num_noise_groups {num_noise_groups}"
        assert (
            num_inference_steps % num_noise_groups == 0
        ), "total inference step number should be divisible by num_noise_groups"
            
        # print(f"Total video tokens in the queue (F): {num_frames}")
        # print(f"Noise group number (G): {num_noise_groups}")
        # print(f"Window size (W): {window_size}")
        # print(f"Num_sample_groups (S): {num_sample_groups}")
        # print(f"Output frame number (S*W*4 + 1): {num_frames * num_noise_groups * 4 + 1}") 

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            height,
            width,
            negative_prompt,
            callback_on_step_end_tensor_inputs,
            prompt_embeds,
            negative_prompt_embeds,
        )
        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._interrupt = False

        # 2. Default call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]
        gen_config.batch_size = batch_size
        device = self._execution_device

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps, **kwargs)

        self._num_timesteps = len(timesteps)

        timesteps_grouped = expand_timesteps_with_group(
            timesteps=timesteps,
            num_frames=num_frames,
            num_noise_groups=num_noise_groups,
            pad_cond_timesteps=with_frame_cond,
            pad_prev_timesteps=False,
        )  # [T//G , F'], where F'=W*G
        gen_config.timesteps_grouped = timesteps_grouped
        
        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 7. Create rotary embeds if required
        image_rotary_emb = (
            self._prepare_rotary_positional_embeddings(height, width, latents.size(1), device)
            if self.transformer.config.use_rotary_positional_embeddings
            else None
        )

        # 8. Denoising loop
        num_warmup_steps = 0

        # gen_config.latents = latents 
        do_classifier_free_guidance = do_classifier_free_guidance and (guidance_scale > 1.0)
        gen_step_config = GenerationStepConfig(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            timesteps_grouped=timesteps_grouped,
            image_rotary_emb=image_rotary_emb,
            window_size=window_size,
            inner_steps=inner_steps,
            batch_size=batch_size,
            num_noise_groups=num_noise_groups,
            num_warmup_steps=num_warmup_steps,
            with_frame_cond=with_frame_cond,
            lcm_multiplier=lcm_multiplier,
            do_classifier_free_guidance=do_classifier_free_guidance,
            attention_kwargs=attention_kwargs,
            extra_step_kwargs=extra_step_kwargs,
        )
        decode_config = DecodeConfig(
            output_type=output_type,
            return_dict=return_dict,
        )
        gen_config.gen_step_config = gen_step_config
        gen_config.decode_config = decode_config
        
        return gen_config
    
    def init_action_window(self, action, n_latents_in_window):
        # This should align with the inital actions to generate the base_video
        self.action_window = [action] * n_latents_in_window
        return self.action_window
    
    def expand_action_window(self, window_size=1, expand_factor=1, pad_action="D"):
        assert len(self.action_window) != 0, "action_window should be initialized before expanding"
        assert pad_action in self.control_signal_to_idx.keys(), f"pad_action should be one of {self.control_signal_to_idx.keys()}"
        expanded_action_window = self.action_window + [pad_action] * window_size * expand_factor

        assert len(expanded_action_window) == self.gen_config.num_frames + self.gen_step_config.window_size * self.gen_step_config.num_noise_groups, "action_window should be expanded before moving"
        return expanded_action_window
    
    def move_action_window(self, new_action, window_size=1):
        assert len(self.action_window) != 0, "action_window should be initialized before moving"
        self.action_window = self.action_window[window_size:] + [new_action] * window_size
        return self.action_window
    
    def update_prompt_embeds(self, new_prompt_embeds):  # deprecated
        self.prompt_embeds = new_prompt_embeds
        return self.prompt_embeds
    
    def send_latents(self, latents_pop, group_idx, gen_config):
        if self.rank == 0:
            with timer(label=f"[RANK {self.rank}]: Moving latents to CPU"):
                latents_pop_cpu = latents_pop.to('cpu')
            with timer(label=f"[RANK {self.rank}]: Sending latents to queue"):
                batch_timestamps = self.pop_timestamps(gen_config.gen_step_config.window_size, gen_config.with_frame_cond)
                self.send_latents_to_queue(latents_pop_cpu, batch_timestamps=batch_timestamps)
            with timer(label=f"[RANK {self.rank}]: Waiting"):
                while(True):
                    vae_step = ray.get(self.vae_step_var.get.remote())
                    if(vae_step < group_idx - 1):
                        print(f"DIT PAUSED: DITSTEP {group_idx}, VAESTEP {vae_step}")
                        self.wait(group_idx, sec=0.003)
                    else:
                        break
            # self.wait(group_idx, sec=gen_config.wait_vae_seconds)  # wait for the preparation of VAE
    
    def get_cur_state(self):
        cur_state = ray.get(self.current_state_var.get.remote())
        return cur_state
    
    def get_cur_prompt(self):
        cur_prompt = ray.get(self.current_prompt_var.get.remote())
        return cur_prompt
    
    def update_cur_prompt(self, new_prompt):
        ray.get(self.current_prompt_var.set.remote(new_prompt))
        return new_prompt
    
    def prepare_init_latents_from_base_video(self, gen_config: GenerationConfig):
        init_video_path, exist = fetch_base_video_path_by_prompt(gen_config.prompt, gen_config.video_cache_dir)
        if exist is False:
            init_video_path = gen_config.default_video_path
            print(f"Fallback to default base video due to no cached base video found for prompt: {gen_config.prompt}")
        gen_config.init_video = load_init_video_clip(init_video_path, gen_config.num_frames, gen_config.fps)
        gen_config.latents = self.prepare_latents_from_video(gen_config)
        return gen_config
    
    def reset_all_ray_var(self):
        ray.get(self.dit_step_var.set.remote(new_value=0))
        ray.get(self.vae_step_var.set.remote(new_value=0))
        ray.get(self.action_manager.get_all.remote())
        ray.get(self.queue_manager.get_all.remote())
        ray.get(self.vae2post_queue.get_all.remote())
        ray.get(self.post2front_queue.get_all.remote())
        
    @torch.no_grad()
    def start_interactive_loop(self, gen_config: GenerationConfig):
        # --> if start from scratch, we need to re-encode the prompt, initialize the latents from init base video clip
        # gen_config.prompt = "xxx"
        gen_config = self.prepare_init_latents_from_base_video(gen_config)
        gen_config = self.prepare_prompt_embeds(gen_config)  # update the prompt_embeds and negative_prompt_embeds in gen_config
        
        # the init_latents are already prepared in the `gen_config.latents`
        latents = gen_config.latents.clone() if gen_config.latents is not None else None
        print("Starting streaming video prediction...")
        for group_idx in range(gen_config.num_sample_groups):
            # --> if prompt is changed, we need to re-encode the prompt
            if group_idx == 100:
                gen_config.prompt = "a driving car is surrounded by many tall trees, and the sun is setting"
                gen_config = self.prepare_prompt_embeds(gen_config)
            if group_idx == 200:
                gen_config.prompt = "a car is driving in a daytime desert with very bright sunshine"
                gen_config = self.prepare_prompt_embeds(gen_config)
            self.print(f"Computing the {group_idx}th/{gen_config.num_sample_groups} group of video tokens...")
            
            group_start_time = time.time()
            # ============= Get Control Signal =====================
            self.print(f"[Group {group_idx}/{gen_config.num_sample_groups}] Receiving control signal...")
            with timer(label=f"[RANK {self.rank}]: Receiving control signal"):
                control_emb = self.get_control_from_signal_interactive(gen_config)
            
            # ================= DiT Latent Generation =================
            latents, latents_pop = self.generate_step_interactive(latents, control_emb, group_idx, gen_config)
                        
            # ================= Send latents_pop =================
            self.send_latents(latents_pop, group_idx, gen_config)

            torch.distributed.barrier()
            torch.cuda.synchronize()
            group_end_time = time.time()
            self.print(f"Group_{group_idx} time: {group_end_time - group_start_time}")
            
        # Offload all models
        self.maybe_free_model_hooks()
        
    @torch.no_grad()
    def start_interactive_loop_(self, gen_config: GenerationConfig):
        # --> if start from scratch, we need to re-encode the prompt, initialize the latents from init base video clip
        # gen_config.prompt = "xxx"
        if self.get_cur_prompt() is None:
            self.update_cur_prompt(gen_config.prompt)
        while True:
            if self.get_cur_state() == "STOP":
                self.reset_all_ray_var()
                time.sleep(0.1)
                continue
            gen_config = self.prepare_init_latents_from_base_video(gen_config)
            gen_config = self.prepare_prompt_embeds(gen_config)  # update the prompt_embeds and negative_prompt_embeds in gen_config
            
            # the init_latents are already prepared in the `gen_config.latents`
            latents = gen_config.latents.clone() if gen_config.latents is not None else None
            print("Starting streaming video prediction...")
            for group_idx in range(gen_config.num_sample_groups):
                cur_state = self.get_cur_state()
                if cur_state == "STOP":
                    break
                # --> if prompt is changed, we need to re-encode the prompt
                cur_prompt = self.get_cur_prompt()
                if cur_prompt != gen_config.prompt:
                    print(f"Prompt changed from \n {gen_config.prompt}\nto\n{cur_prompt}")
                    gen_config.prompt = self.get_cur_prompt()
                    gen_config = self.prepare_prompt_embeds(gen_config)
                    
                self.print(f"Computing the {group_idx}th/{gen_config.num_sample_groups} group of video tokens...")
                
                group_start_time = time.time()
                # ============= Get Control Signal =====================
                self.print(f"[Group {group_idx}/{gen_config.num_sample_groups}] Receiving control signal...")
                with timer(label=f"[RANK {self.rank}]: Receiving control signal"):
                    control_emb = self.get_control_from_signal_interactive(gen_config)
                
                # ================= DiT Latent Generation =================
                latents, latents_pop = self.generate_step_interactive(latents, control_emb, group_idx, gen_config)
                            
                # ================= Send latents_pop =================
                self.send_latents(latents_pop, group_idx, gen_config)

                torch.distributed.barrier()
                torch.cuda.synchronize()
                group_end_time = time.time()
                self.print(f"Group_{group_idx} time: {group_end_time - group_start_time}")
                
            # Offload all models
            self.maybe_free_model_hooks()