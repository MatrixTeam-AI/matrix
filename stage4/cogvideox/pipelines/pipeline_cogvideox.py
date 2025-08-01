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
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from transformers import T5EncoderModel, T5Tokenizer

from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.models.embeddings import get_3d_rotary_pos_embed
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.utils import logging, replace_example_docstring
from diffusers.utils.torch_utils import randn_tensor
from diffusers.video_processor import VideoProcessor

from ..loader import CogVideoXLoraLoaderMixin
from ..autoencoder import AutoencoderKLCogVideoX
from ..transformer import CogVideoXTransformer3DModel
from ..scheduler import (
    LCMSwinScheduler,
    CogVideoXDPMScheduler,
    CogVideoXSwinDPMScheduler,
    expand_timesteps_with_group,
)
from .pipeline_output import CogVideoXPipelineOutput
from ..control_adapter import CONTROL_SIGNAL_TO_PROMPT

from transformers import AutoTokenizer, CLIPModel

from contextlib import nullcontext

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


EXAMPLE_DOC_STRING = """
    Examples:
        ```python
        >>> import torch
        >>> from diffusers import CogVideoXPipeline
        >>> from diffusers.utils import export_to_video

        >>> # Models: "THUDM/CogVideoX-2b" or "THUDM/CogVideoX-5b"
        >>> pipe = CogVideoXPipeline.from_pretrained("THUDM/CogVideoX-2b", torch_dtype=torch.float16).to("cuda")
        >>> prompt = (
        ...     "A panda, dressed in a small, red jacket and a tiny hat, sits on a wooden stool in a serene bamboo forest. "
        ...     "The panda's fluffy paws strum a miniature acoustic guitar, producing soft, melodic tunes. Nearby, a few other "
        ...     "pandas gather, watching curiously and some clapping in rhythm. Sunlight filters through the tall bamboo, "
        ...     "casting a gentle glow on the scene. The panda's face is expressive, showing concentration and joy as it plays. "
        ...     "The background includes a small, flowing stream and vibrant green foliage, enhancing the peaceful and magical "
        ...     "atmosphere of this unique musical performance."
        ... )
        >>> video = pipe(prompt=prompt, guidance_scale=6, num_inference_steps=50).frames[0]
        >>> export_to_video(video, "output.mp4", fps=8)
        ```
"""


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


class CogVideoXStreamingPipeline(CogVideoXPipeline):
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
    
    def get_control_from_signal(self, control_signal, start=None, end=None):
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

        control_signal_list = control_signal.split(",")
        if start is not None and end is not None:
            control_signal_list = control_signal_list[start: end]
        control_indices = [self.control_signal_to_idx[c] for c in control_signal_list]
        control = self.control_embeddings[control_indices]
        return control

    
    def decode_latents(self, latents: torch.Tensor, sliced_decode=False) -> torch.Tensor:
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
                cur_frames = self.vae.decode(latents[:, :, cur_start:cur_start+decode_window]).sample
                cur_frames_cpu = cur_frames.cpu()
                del cur_frames
                all_frames.append(cur_frames_cpu)
                cur_start += (decode_window - overlap_length)
            if cur_start + overlap_length < num_latent_frames or cur_start == 0:
                cur_frames = self.vae.decode(latents[:, :, cur_start:]).sample
                cur_frames_cpu = cur_frames.cpu()
                del cur_frames
                all_frames.append(cur_frames_cpu)
            frames = [all_frames[0]]
            frames.extend([item[:, :, overlap_length*4:] for item in all_frames[1:]])
            frames = torch.cat(frames, dim=2)
        else:
            frames = self.vae.decode(latents).sample
        
        return frames

    @torch.no_grad()
    def __call__(
        self,
        init_video=None,  # add init video for video prediction, frame number = 49
        prompt=None,
        negative_prompt=None,
        height=480,
        width=720,
        num_frames=49,
        num_inference_steps=20,
        timesteps=None,
        guidance_scale=6,
        use_dynamic_cfg=False,
        num_videos_per_prompt=1,
        eta=0,
        generator=None,
        latents=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        output_type="pil",
        return_dict=True,
        attention_kwargs=None,
        callback_on_step_end_tensor_inputs=["latents"],
        max_sequence_length=226,
        control_signal: List[str] = None,
        num_noise_groups=4,  # number of noise group number
        num_sample_groups=8,  # number of outer sampling loop, and each iteration output a group of video tokens.
        with_frame_cond=True,  # whether to use frame condition, if ture the first frame is used as condition with zero noise. MUST BE TRUE here.
        do_classifier_free_guidance=False,
        **kwargs
    ):

        # Total sampling steps = num_inference_steps * num_sample_groups // num_noise_group
        # Outer sampling steps = num_sample_groups
        # Inner sampling steps = num_inference_steps // num_noise_group

        # if num_frames > 49:
        #     raise ValueError(
        #         "The number of frames must be less than 49 for now due to static positional embeddings. This will be updated in the future to remove this limitation."
        #     )
        lcm_multiplier = kwargs.pop('lcm_multiplier', 1)

        if with_frame_cond:
            num_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1 if latents is None else latents.size(1)
            num_videos_per_prompt = 1
            outer_steps = num_sample_groups
            inner_steps = num_inference_steps // num_noise_groups
            window_size = (num_frames - 1) // num_noise_groups
        else:
            num_frames = num_frames // self.vae_scale_factor_temporal if latents is None else latents.size(1)
            num_videos_per_prompt = 1
            outer_steps = num_sample_groups
            inner_steps = num_inference_steps // num_noise_groups
            window_size = num_frames // num_noise_groups
        
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

        device = self._execution_device

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
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

        # 5. Prepare latents.
        if latents is None:
            init_video = self.video_processor.preprocess_video(init_video, height=height, width=width)
            init_video = init_video.to(device=device, dtype=prompt_embeds.dtype)
            latents = None
        else:
            init_video = None
            latents = latents.to(device, dtype=prompt_embeds.dtype)

        latent_channels = self.transformer.config.in_channels
        latents = self.prepare_latents(
            init_video,
            batch_size * num_videos_per_prompt,
            latent_channels,
            num_frames,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
            timesteps_grouped[0],
        )
        print("Latent size:", latents.size())
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

        # Control signal
        control_start = 0 if with_frame_cond else 1

        print("Starting streaming video prediction...")
        latents_pop_stream = []
        for group_idx in range(outer_steps):
            print(f"Finished group number is {len(latents_pop_stream)}...Computing for the {group_idx + 1}th/{num_sample_groups} group of video tokens...")

            # Control signal
            control_emb = self.get_control_from_signal(control_signal, control_start, control_start+num_frames)
            control_emb = control_emb.unsqueeze(0).to(prompt_embeds.dtype).contiguous()
            control_start += window_size
            
            if do_classifier_free_guidance:
                control_emb = torch.cat([control_emb, control_emb], dim=0)

            with self.progress_bar(total=inner_steps) as progress_bar:
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
                    if torch.backends.mps.is_available():
                        autocast_ctx = nullcontext()
                    else:
                        autocast_ctx = torch.autocast("cuda", dtype=latents.dtype)

                    with autocast_ctx:
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


                    if do_classifier_free_guidance:
                        guidance_scale = guidance_scale * torch.ones(batch_size)
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                    # compute the previous noisy sample x_t -> x_t-1
                    if isinstance(self.scheduler, CogVideoXSwinDPMScheduler):
                        latents_cond = latents[:, :1]
                        latents, old_pred_original_sample = self.scheduler.step(
                            noise_pred[:,1:],
                            old_pred_original_sample,
                            timesteps[:,1:],
                            timesteps_back[:,1:] if timesteps_back is not None else None,
                            latents[:,1:],
                            **extra_step_kwargs,
                            return_dict=False,
                        )
                        latents = latents.to(prompt_embeds.dtype)
                        latents = torch.cat([latents_cond, latents], dim=1) 
                         # keep cond frame unchanged
                    elif isinstance(self.scheduler, LCMSwinScheduler):
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
                        latents = torch.cat([latents_cond, latents], dim=1)
                    else :
                        latents_cond = latents[:, :1]
                        latents, old_pred_original_sample = self.scheduler.step(
                            noise_pred[:,1:],
                            timesteps[:,1:],
                            latents[:,1:],
                            **extra_step_kwargs,
                            return_dict=False,
                        )
                        latents = latents.to(prompt_embeds.dtype)
                        latents = torch.cat([latents_cond, latents], dim=1)  # keep cond frame unchanged 

                    if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                        progress_bar.update()

            if with_frame_cond:
                latents_pop = latents[:, 1 : window_size + 1]
                latents_remain = latents[:, window_size + 1 :]
                latents_cond_new = latents_pop[:, -1].unsqueeze(1)
                latents_new = torch.cat(
                    [latents_cond_new, latents_remain, torch.randn_like(latents_pop)],
                    dim=1,
                )  # append noisy video token to the right of the video sequence
                latents_pop_stream.append(latents_pop)
                latents = latents_new
            else:
                latents_pop = latents[:, :window_size]
                latents_remain = latents[:, window_size:]
                latents_new = torch.cat([latents_remain, torch.randn_like(latents_pop)], dim=1)
                latents_pop_stream.append(latents_pop)
                latents = latents_new

        latents_out = torch.cat(latents_pop_stream, dim=1,)

        if not output_type == "latent":
            video = self.decode_latents(latents_out, sliced_decode=True)
            video = self.video_processor.postprocess_video(video=video, output_type=output_type)
        else:
            video = latents_out

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (video,)

        return CogVideoXPipelineOutput(frames=video)
