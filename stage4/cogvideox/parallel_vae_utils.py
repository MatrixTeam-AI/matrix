# Modified from https://github.com/xdit-project/DistVAE/blob/0.0.0beta5/distvae/utils.py
import os

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed import ProcessGroup

class VAEParallelState:
    _vae_group = None
    _local_rank = None
    _world_size = None
    _vae_rank = None
    _is_split = None

    @classmethod
    def initialize(cls, vae_group: ProcessGroup):
        assert dist.is_available() and dist.is_initialized()
        if vae_group is None:
            cls._vae_group = dist.group.WORLD
        else:
            cls._vae_group = vae_group
        cls._local_rank = int(os.environ.get('LOCAL_RANK', 0)) # FIXME: in ray all local_rank is 0
        cls._rank_mapping = None
        cls._world_size = dist.get_world_size(cls.get_vae_group())
        cls._vae_rank = dist.get_rank(cls.get_vae_group())
        cls._is_split = False
        
        cls._init_rank_mapping()

    @classmethod
    def is_initialized(cls):
        return cls._vae_group is not None

    @classmethod
    def is_split(cls):
        return cls.is_initialized() and cls._is_split

    @classmethod
    def set_split_state(cls, split_state: bool):
        cls._is_split = split_state

    @classmethod
    def get_vae_group(cls) -> ProcessGroup:
        if cls._vae_group is None:
            raise RuntimeError("VAEParallelState not initialized. Call initialize() first.")
        return cls._vae_group

    @classmethod
    def get_global_rank(cls) -> int:
        return dist.get_rank()
    
    @classmethod
    def _init_rank_mapping(cls):
        """Initialize the mapping between group ranks and global ranks"""
        if cls._rank_mapping is None:
            # Get all ranks in the group
            ranks = [None] * cls.get_group_world_size()
            dist.all_gather_object(ranks, cls.get_global_rank(), group=cls.get_vae_group())
            cls._rank_mapping = ranks

    @classmethod
    def get_global_rank_from_group_rank(cls, group_rank: int) -> int:
        """Convert a rank in VAE group to global rank using cached mapping.
        
        Args:
            group_rank: The rank in VAE group
            
        Returns:
            The corresponding global rank
            
        Raises:
            RuntimeError: If the group_rank is invalid
        """
        if cls._rank_mapping is None:
            cls._init_rank_mapping()
            
        if group_rank < 0 or group_rank >= cls.get_group_world_size():
            raise RuntimeError(f"Invalid group rank: {group_rank}. Must be in range [0, {cls.get_group_world_size()-1}]")
            
        return cls._rank_mapping[group_rank]
    
    @classmethod
    def get_rank_in_vae_group(cls) -> int:
        return cls._vae_rank

    @classmethod
    def get_group_world_size(cls) -> int:
        return cls._world_size

    @classmethod
    def get_local_rank(cls) -> int:
        return cls._local_rank


class Patchify(nn.Module):
    def forward(self, hidden_state):
        r"""Split `hidden_state` along the `width` dimension across different GPUs."""
        group_world_size = VAEParallelState.get_group_world_size()
        rank_in_vae_group = VAEParallelState.get_rank_in_vae_group()
        width = hidden_state.shape[-1]

        start_idx = (width + group_world_size - 1) // group_world_size * rank_in_vae_group
        end_idx = min((width + group_world_size - 1) // group_world_size * (rank_in_vae_group + 1), width)

        return hidden_state[:, :, :, :, start_idx : end_idx].clone()


class DePatchify(nn.Module):
    def forward(self, patch_hidden_state):
        r"""Gather `hidden_state` along the `width` dimension across different GPUs."""
        group_world_size = VAEParallelState.get_group_world_size()
        local_rank = VAEParallelState.get_local_rank()
        vae_group = VAEParallelState.get_vae_group()
        width = patch_hidden_state.shape[-1]

        patch_width_list = [
            torch.empty([1], dtype=torch.int64, device=f"cuda:{local_rank}")
            for _ in range(group_world_size)
        ]
        dist.all_gather(
            patch_width_list,
            torch.tensor(
                [width], # width dimension
                dtype=torch.int64, 
                device=f"cuda:{local_rank}"
            ),
            group=vae_group,
        )

        patch_hidden_state_list = [
            torch.empty(
                [*patch_hidden_state.shape[:-1], width], 
                dtype=patch_hidden_state.dtype,
                device=f"cuda:{local_rank}"
            ) for i in range(group_world_size)
        ]
        dist.all_gather(
            patch_hidden_state_list, 
            patch_hidden_state.contiguous(),
            group=vae_group
        )
        return torch.cat(patch_hidden_state_list, dim=-1)