# This file is modified from https://github.com/xdit-project/xDiT/blob/0.4.1/xfuser/model_executor/layers/usp.py
# This file implements USP with torch version >= '2.5.0'
from einops import rearrange
import numpy as np
import torch
from torch.nn import functional as F
import torch.distributed as dist
from torch.distributed.tensor.experimental._attention import _templated_ring_attention
aten = torch.ops.aten

import torch.distributed._functional_collectives as ft_c

from yunchang.globals import PROCESS_GROUP

from xfuser.core.distributed import (
    get_sequence_parallel_world_size,
    get_ulysses_parallel_world_size,
    get_ring_parallel_world_size,
    get_ulysses_parallel_rank,
)

def ring_attn(query, key, value, dropout_p=0.0, is_causal=False):
    out, *_ = _templated_ring_attention(
        PROCESS_GROUP.RING_PG,
        aten._scaled_dot_product_flash_attention,
        query,
        key,
        value,
        dropout_p=dropout_p,
        is_causal=is_causal
    )
    return out


def _maybe_wait(tensor: torch.Tensor) -> torch.Tensor:
    """
    When tracing the code, the result tensor is not an AsyncCollectiveTensor,
    so we cannot call ``wait()``.
    """
    if isinstance(tensor, ft_c.AsyncCollectiveTensor):
        return tensor.wait()
    return tensor


def _sdpa_all_to_all_single(x):
    x_shape = x.shape
    x = x.flatten()
    x = ft_c.all_to_all_single(x, output_split_sizes=None, input_split_sizes=None, group=PROCESS_GROUP.ULYSSES_PG)
    x = _maybe_wait(x)
    x = x.reshape(x_shape)
    return x


def _ft_c_input_all_to_all(x):
    world_size = get_ulysses_parallel_world_size()
    if world_size <= 1:
        return x

    assert x.ndim == 4, "x must have 4 dimensions, got {}".format(x.ndim)
    b, h, s, d = x.shape
    assert h % world_size == 0, "h must be divisible by world_size, got {} and {}".format(h, world_size)

    x = x.permute(1, 0, 2, 3).contiguous()
    x = _sdpa_all_to_all_single(x)
    x = x.reshape(world_size, h // world_size, b, -1, d).permute(2, 1, 0, 3, 4).reshape(b, h // world_size, -1, d)
    return x


def _ft_c_output_all_to_all(x):
    world_size = get_ulysses_parallel_world_size()
    if world_size <= 1:
        return x

    assert x.ndim == 4, "x must have 4 dimensions, got {}".format(x.ndim)
    b, h, s, d = x.shape
    assert s % world_size == 0, "s must be divisible by world_size, got {} and {}".format(s, world_size)

    x = x.permute(2, 0, 1, 3).contiguous()
    x = _sdpa_all_to_all_single(x)
    x = x.reshape(world_size, s // world_size, b, -1, d).permute(2, 0, 3, 1, 4).reshape(b, -1, s // world_size, d)
    return x

def _input_split(x, dim=1):
    world_size = get_ulysses_parallel_world_size()
    if world_size <= 1:
        return x

    assert x.ndim == 4, "x must have 4 dimensions, got {}".format(x.ndim)
    assert x.shape[dim] % world_size == 0, "h must be divisible by world_size, got {} and {}".format(x.shape[dim], world_size)

    x = x.chunk(world_size, dim=1)[get_ulysses_parallel_rank()].contiguous()
    return x

def USP(
    query, key, value, dropout_p=0.0, is_causal=False, 
    seq_length=None
):
    if get_sequence_parallel_world_size() == 1:
        out = F.scaled_dot_product_attention(
            query, key, value, dropout_p=dropout_p, is_causal=is_causal
        )
    elif get_ulysses_parallel_world_size() == 1:
        raise NotImplementedError()
        out = ring_attn(query, key, value, dropout_p=dropout_p, is_causal=is_causal)
    elif get_ulysses_parallel_world_size() > 1:
        input_all_to_all_func = _ft_c_input_all_to_all
        output_all_to_all_func = _ft_c_output_all_to_all

        query = input_all_to_all_func(query)
        key = input_all_to_all_func(key)
        value = input_all_to_all_func(value)

        #remove padded zeros
        #shape: [batch_size, num_heads, seq_length, num_channels]
        original_length = query.shape[-2]
        if seq_length is not None and seq_length < original_length:
            query = query[:, :, -seq_length : ]
            key = key[:, :, -seq_length : ]
            value = value[:, :, -seq_length : ]

        if get_ring_parallel_world_size() == 1:
            out = F.scaled_dot_product_attention(
                query, key, value, dropout_p=dropout_p, is_causal=is_causal
            )
        else:
            out = ring_attn(query, key, value, dropout_p=dropout_p, is_causal=is_causal)

        #padding zeros
        if seq_length is not None and seq_length < original_length:
            b, h, s, d = out.shape
            assert seq_length == s
            padding_length = original_length - s
            out = torch.cat([
                out.new_zeros([b, h, padding_length, d]),
                out,
            ], dim=-2)

        out = output_all_to_all_func(out)
        
    return out


def custom_USP(
    query, key, value, dropout_p=0.0, is_causal=False, 
    key_ops: str = "split,repeat", 
    value_ops: str = "split,repeat", 
    temporal_length: int = None
):
    ulysses_size = get_ulysses_parallel_world_size()
    
    if get_sequence_parallel_world_size() == 1:
        b, h, n, d = query.shape
        key = key.repeat_interleave(n // key.shape[2], dim=2)
        assert key.shape[2] == n
        value = value.repeat_interleave(n // value.shape[2], dim=2)
        assert value.shape[2] == n

        out = F.scaled_dot_product_attention(
            query, key, value, dropout_p=dropout_p, is_causal=is_causal
        )

    elif ulysses_size == 1:
        raise NotImplementedError()
        out = ring_attn(query, key, value, dropout_p=dropout_p, is_causal=is_causal)

    elif ulysses_size > 1:
        query = _ft_c_input_all_to_all(query)
        b, h, n, d = query.shape
        if temporal_length is not None:
            query = rearrange(
                query, 
                "B H (U T S) D -> B H (T U S) D",
                U = ulysses_size,
                T = temporal_length,
                S = n // (ulysses_size * temporal_length)
            )

        for op in key_ops.split(','):
            if op == 'repeat':
                key = key.repeat_interleave(n // key.shape[2], dim=2)
            elif op == 'split':
                key = _input_split(key, dim=1)
            elif op == 'all-to-all':
                key = _ft_c_input_all_to_all(key)
            else:
                raise NotImplementedError()
        assert key.shape[2] == n

        for op in value_ops.split(','):
            if op == 'repeat':
                value = value.repeat_interleave(n // value.shape[2], dim=2)
            elif op == 'split':
                value = _input_split(value, dim=1)
            elif op == 'all-to-all':
                value = _ft_c_input_all_to_all(value)
            else:
                raise NotImplementedError()
        assert value.shape[2] == n

        if get_ring_parallel_world_size() == 1:
            out = F.scaled_dot_product_attention(
                query, key, value, dropout_p=dropout_p, is_causal=is_causal
            )
        else:
            out = ring_attn(query, key, value, dropout_p=dropout_p, is_causal=is_causal)

        out = _ft_c_output_all_to_all(out)
        
    return out