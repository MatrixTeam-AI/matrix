# This file is modified from https://github.com/xdit-project/xDiT/blob/0.4.1/xfuser/model_executor/layers/attention_processor.py
import inspect
from typing import Optional

import torch
from torch import nn
import torch.distributed
from torch.nn import functional as F
from diffusers.utils import deprecate
from diffusers.models.attention import Attention
from diffusers.models.embeddings import apply_rotary_emb

from xfuser.core.distributed import (
    get_sequence_parallel_world_size,
    get_pipeline_parallel_world_size,
)
from xfuser.core.fast_attention import (
    xFuserFastAttention,
    get_fast_attn_enable,
)
from xfuser.core.cache_manager.cache_manager import get_cache_manager
from xfuser.core.distributed.runtime_state import get_runtime_state
from xfuser.model_executor.layers import xFuserLayerBaseWrapper
from xfuser.model_executor.layers import xFuserLayerWrappersRegister
from xfuser.logger import init_logger
from xfuser.envs import PACKAGES_CHECKER
if torch.__version__ >= "2.5.0":
    from .usp import USP, custom_USP
else:
    from xfuser.model_executor.layers.usp_legacy import USP

from ..attention_processor import CogVideoXAttnProcessor2_0
from ..control_adapter import CogVideoXControlAttnProcessor2_0

logger = init_logger(__name__)

env_info = PACKAGES_CHECKER.get_packages_info()
HAS_LONG_CTX_ATTN = env_info["has_long_ctx_attn"]
HAS_FLASH_ATTN = env_info["has_flash_attn"]

def is_v100():
    if not torch.cuda.is_available():
        return False
    device_name = torch.cuda.get_device_name(torch.cuda.current_device())
    return "V100" in device_name


def torch_compile_disable_if_v100(func):
    if is_v100():
        return torch.compiler.disable(func)
    return func


class xFuserAttentionBaseWrapper(xFuserLayerBaseWrapper):
    def __init__(
        self,
        attention: Attention,
    ):
        super().__init__(module=attention)

        to_k = self.module.to_k
        to_v = self.module.to_v
        assert isinstance(to_k, nn.Linear)
        assert isinstance(to_v, nn.Linear)
        assert (to_k.bias is None) == (to_v.bias is None)
        assert to_k.weight.shape == to_v.weight.shape


class xFuserAttentionProcessorRegister:
    _XFUSER_ATTENTION_PROCESSOR_MAPPING = {}

    @classmethod
    def register(cls, origin_processor_class):
        def decorator(xfuser_processor):
            if not issubclass(xfuser_processor, origin_processor_class):
                raise ValueError(
                    f"{xfuser_processor.__class__.__name__} is not a subclass of origin class {origin_processor_class.__class__.__name__}"
                )
            cls._XFUSER_ATTENTION_PROCESSOR_MAPPING[origin_processor_class] = (
                xfuser_processor
            )
            return xfuser_processor

        return decorator

    @classmethod
    def get_processor(cls, processor):
        for (
            origin_processor_class,
            xfuser_processor,
        ) in cls._XFUSER_ATTENTION_PROCESSOR_MAPPING.items():
            if isinstance(processor, origin_processor_class):
                return xfuser_processor
        raise ValueError(
            f"Attention Processor class {processor.__class__.__name__} is not supported by xFuser"
        )


@xFuserLayerWrappersRegister.register(Attention)
class xFuserAttentionWrapper(xFuserAttentionBaseWrapper):
    def __init__(
        self,
        attention: Attention,
        latte_temporal_attention: bool = False,
    ):
        super().__init__(attention=attention)
        self.processor = xFuserAttentionProcessorRegister.get_processor(
            attention.processor
        )()
        self.latte_temporal_attention = latte_temporal_attention

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **cross_attention_kwargs,
    ) -> torch.Tensor:
        r"""
        The forward method of the `Attention` class.

        Args:
            hidden_states (`torch.Tensor`):
                The hidden states of the query.
            encoder_hidden_states (`torch.Tensor`, *optional*):
                The hidden states of the encoder.
            attention_mask (`torch.Tensor`, *optional*):
                The attention mask to use. If `None`, no mask is applied.
            **cross_attention_kwargs:
                Additional keyword arguments to pass along to the cross attention.

        Returns:
            `torch.Tensor`: The output of the attention layer.
        """
        # The `Attention` class can call different attention processors / attention functions
        # here we simply pass along all tensors to the selected processor class
        # For standard processors that are defined here, `**cross_attention_kwargs` is empty
        attn_parameters = set(
            inspect.signature(self.processor.__call__).parameters.keys()
        )
        quiet_attn_parameters = {"ip_adapter_masks"}
        unused_kwargs = [
            k
            for k, _ in cross_attention_kwargs.items()
            if k not in attn_parameters and k not in quiet_attn_parameters
        ]
        if len(unused_kwargs) > 0:
            logger.warning(
                f"cross_attention_kwargs {unused_kwargs} are not expected by {self.processor.__class__.__name__} and will be ignored."
            )
        cross_attention_kwargs = {
            k: w for k, w in cross_attention_kwargs.items() if k in attn_parameters
        }

        return self.processor(
            self,
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            latte_temporal_attention=self.latte_temporal_attention,
            **cross_attention_kwargs,
        )

@xFuserAttentionProcessorRegister.register(CogVideoXAttnProcessor2_0)
class xFuserCogVideoXAttnProcessor2_0(CogVideoXAttnProcessor2_0):
    r"""
    Processor for implementing scaled dot-product attention for the CogVideoX model. It applies a rotary embedding on
    query and key vectors, but does not include spatial normalization.
    """

    def __init__(self, seq_length=226):
        super().__init__()
        self.seq_length = seq_length
        use_long_ctx_attn_kvcache = True
        self.use_long_ctx_attn_kvcache = (
            HAS_LONG_CTX_ATTN
            and use_long_ctx_attn_kvcache
            and get_sequence_parallel_world_size() > 1
        )
        if HAS_LONG_CTX_ATTN and get_sequence_parallel_world_size() > 1:
            from xfuser.core.long_ctx_attention import (
                xFuserLongContextAttention,
            )

            if HAS_FLASH_ATTN:
                self.hybrid_seq_parallel_attn = xFuserLongContextAttention(
                    use_kv_cache=self.use_long_ctx_attn_kvcache
                )
            else:
                from yunchang.kernels import AttnType

                self.hybrid_seq_parallel_attn = xFuserLongContextAttention(
                    use_kv_cache=self.use_long_ctx_attn_kvcache,
                    attn_type=AttnType.TORCH,
                )
        else:
            self.hybrid_seq_parallel_attn = None

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        text_seq_length = encoder_hidden_states.size(1)
        latent_seq_length = hidden_states.size(1)

        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        batch_size, sequence_length, _ = (
            hidden_states.shape
            if encoder_hidden_states is None
            else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(
                attention_mask, sequence_length, batch_size
            )
            attention_mask = attention_mask.view(
                batch_size, attn.heads, -1, attention_mask.shape[-1]
            )

        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Apply RoPE if needed
        if image_rotary_emb is not None:
            query[:, :, text_seq_length:] = apply_rotary_emb(
                query[:, :, text_seq_length:], image_rotary_emb
            )
            if not attn.is_cross_attention:
                key[:, :, text_seq_length:] = apply_rotary_emb(
                    key[:, :, text_seq_length:], image_rotary_emb
                )

        #! ---------------------------------------- ATTENTION ----------------------------------------
        if (
            get_pipeline_parallel_world_size() == 1
            and get_runtime_state().split_text_embed_in_sp
        ):
            hidden_states = USP(query, key, value, dropout_p=0.0, is_causal=False, seq_length=self.seq_length)
            hidden_states = hidden_states.transpose(1, 2).reshape(
                batch_size, -1, attn.heads * head_dim
            )
        elif HAS_LONG_CTX_ATTN and get_sequence_parallel_world_size() > 1:
            if get_runtime_state().split_text_embed_in_sp:
                encoder_query = None
                encoder_key = None
                encoder_value = None
            else:
                encoder_query = query[:, :, :text_seq_length, :]
                query = query[:, :, text_seq_length:, :]
                encoder_key = key[:, :, :text_seq_length, :]
                key = key[:, :, text_seq_length:, :]
                encoder_value = value[:, :, :text_seq_length, :]
                value = value[:, :, text_seq_length:, :]

                encoder_query = encoder_query.transpose(1, 2)
                encoder_key = encoder_key.transpose(1, 2)
                encoder_value = encoder_value.transpose(1, 2)

            query = query.transpose(1, 2)
            key = key.transpose(1, 2)
            value = value.transpose(1, 2)

            hidden_states = self.hybrid_seq_parallel_attn(
                None,
                query,
                key,
                value,
                dropout_p=0.0,
                causal=False,
                joint_tensor_query=encoder_query,
                joint_tensor_key=encoder_key,
                joint_tensor_value=encoder_value,
                joint_strategy="front",
            )

            hidden_states = hidden_states.reshape(batch_size, -1, attn.heads * head_dim)
        else:
            if HAS_FLASH_ATTN:
                from flash_attn import flash_attn_func

                query = query.transpose(1, 2)
                key = key.transpose(1, 2)
                value = value.transpose(1, 2)
                hidden_states = flash_attn_func(
                    query, key, value, dropout_p=0.0, causal=False
                )
                hidden_states = hidden_states.reshape(
                    batch_size, -1, attn.heads * head_dim
                )

            else:
                # the output of sdp = (batch, num_heads, seq_len, head_dim)
                # TODO: add support for attn.scale when we move to Torch 2.1
                hidden_states = F.scaled_dot_product_attention(
                    query, key, value, dropout_p=0.0, is_causal=False
                )
                hidden_states = hidden_states.transpose(1, 2).reshape(
                    batch_size, -1, attn.heads * head_dim
                )

        #! ORIGIN
        # hidden_states = F.scaled_dot_product_attention(
        #     query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        # )
        # hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        #! ---------------------------------------- ATTENTION ----------------------------------------

        assert text_seq_length + latent_seq_length == hidden_states.shape[1]
        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        encoder_hidden_states, hidden_states = hidden_states.split(
            [text_seq_length, latent_seq_length], dim=1
        )
        return hidden_states, encoder_hidden_states

@xFuserAttentionProcessorRegister.register(CogVideoXControlAttnProcessor2_0)
class xFuserCogVideoXControlAttnProcessor2_0(CogVideoXControlAttnProcessor2_0):
    r"""
    Processor for implementing scaled dot-product attention for the CogVideoX model. It applies a rotary embedding on
    query and key vectors, but does not include spatial normalization.
    """

    def __init__(self, temporal_length: int = None):
        super().__init__()
        self.temporal_length = temporal_length
        use_long_ctx_attn_kvcache = True
        self.use_long_ctx_attn_kvcache = (
            HAS_LONG_CTX_ATTN
            and use_long_ctx_attn_kvcache
            and get_sequence_parallel_world_size() > 1
        )
        if HAS_LONG_CTX_ATTN and get_sequence_parallel_world_size() > 1:
            from xfuser.core.long_ctx_attention import (
                xFuserLongContextAttention,
            )

            if HAS_FLASH_ATTN:
                self.hybrid_seq_parallel_attn = xFuserLongContextAttention(
                    use_kv_cache=self.use_long_ctx_attn_kvcache
                )
            else:
                from yunchang.kernels import AttnType

                self.hybrid_seq_parallel_attn = xFuserLongContextAttention(
                    use_kv_cache=self.use_long_ctx_attn_kvcache,
                    attn_type=AttnType.TORCH,
                )
        else:
            self.hybrid_seq_parallel_attn = None

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        batch_size, sequence_length, _ = hidden_states.shape
        _, encoder_sequence_length, _ = encoder_hidden_states.shape
        assert sequence_length % encoder_sequence_length == 0

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(
                attention_mask, sequence_length, batch_size
            )
            attention_mask = attention_mask.view(
                batch_size, attn.heads, -1, attention_mask.shape[-1]
            )

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Apply RoPE if needed
        if image_rotary_emb is not None:
            raise NotImplementedError()
            query = apply_rotary_emb(query, image_rotary_emb)
            if not attn.is_cross_attention:
                key = apply_rotary_emb(key, image_rotary_emb)

        #! ---------------------------------------- ATTENTION ----------------------------------------
        if get_pipeline_parallel_world_size() == 1:
            hidden_states = custom_USP(
                query, key, value, dropout_p=0.0, is_causal=True, 
                key_ops="split,repeat", value_ops="split,repeat", temporal_length=self.temporal_length
            )
            hidden_states = hidden_states.transpose(1, 2).reshape(
                batch_size, -1, attn.heads * head_dim
            )
        else:
            raise NotImplementedError()

        #! ORIGIN
        # hidden_states = F.scaled_dot_product_attention(
        #     query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=True
        # )
        # hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        #! ---------------------------------------- ATTENTION ----------------------------------------

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states