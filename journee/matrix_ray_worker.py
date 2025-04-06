import os
import sys
import time
from dataclasses import dataclass
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple
from itertools import islice, repeat

from PIL import Image
import torch
import torch.distributed as dist
import numpy as np
from diffusers.video_processor import VideoProcessor
from diffusers.utils import export_to_video, load_image, load_video

import ray
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from xfuser.ray.pipeline.ray_utils import initialize_ray_cluster
from xfuser.ray.pipeline.pipeline_utils import GPUExecutor
from xfuser.ray.worker.worker_wrappers import RayWorkerWrapper
from xfuser.ray.worker.worker import WorkerBase
from xfuser.core.distributed.parallel_state import (init_distributed_environment,
                                                    init_vae_group,
                                                    get_world_group,
                                                    get_vae_parallel_group)

from ray_pipeline_utils import timer, add_timestamp, get_data_and_passed_time
sys.path.insert(0, '/'.join(os.path.realpath(__file__).split('/')[:-2]))
from stage3.cogvideox.autoencoder import AutoencoderKLCogVideoX
from stage3.cogvideox.parallel_vae_utils import VAEParallelState
try:
    from post_processors.Interpolator_RIFE import RIFEInterpolator
except ImportError as e:
    print(f"ImportError: {e}")

@dataclass
class ParallelConfig:
    world_size: int = 1
    dit_parallel_size: int = 0
    vae_parallel_size: int = 1
    post_parallel_size: int = 1
    def __post_init__(self):
        self.dp_degree = 1
        self.cfg_degree = 1
        self.sp_degree = 1
        self.pp_degree = 1
        self.tp_degree = 1
    
@dataclass(frozen=True)
class EngineConfig:
    parallel_config: ParallelConfig
    # model_config: ModelConfig
    # runtime_config: RuntimeConfig
    # fast_attn_config: FastAttnConfig

def frames_pt_to_numpy_list(frames_pt: torch.Tensor) -> List[np.ndarray]:
    return [frame for frame in frames_pt.cpu().numpy()]
    
class ParallelVAEWrapper:
    def __init__(
        self, 
        vae,
    ):
        self.vae = vae
        self.video_processor = VideoProcessor(vae_scale_factor=8)
    
    @torch.no_grad()
    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        # latents.shape: [batch_size, num_latents, num_channels=16, height, width]
        # frames.shape: [batch_size, num_channels=3, num_frames, height, width]
        vae_scaling_factor_image = self.vae.config.scaling_factor
    
        assert latents.device == self.vae.device
        latents = latents.permute(0, 2, 1, 3, 4)  # [batch_size, num_frames, num_channels, height, width]
        latents = 1 / vae_scaling_factor_image * latents
        frames = self.vae.decode(latents, keep_cache=True).sample
        # post process
        assert frames.shape[0] == 1, "Batch size should be 1"
        frames = self.video_processor.postprocess(
            frames[0].permute(1, 0, 2, 3), # [C, F, H ,W] -> [F, C, H, W]
            output_type="pt",
        )
        # [F, C, H, W] -> [F, H, W, C] and denormalize
        frames = (frames.permute(0, 2, 3, 1).float() * 255).to(torch.uint8)
        return frames

    def execute(self, **kwargs):
        latents = kwargs.get('latents', None)
        assert latents is not None and self.vae is not None
        rank = get_world_group().rank
        # print(f"Rank {rank} is running the VAE")
        latents = latents.to(self.vae.device)
        print(f"(RANK {rank})input of vae worker: ", latents.shape)
        frames = self.decode_latents(latents)
        return frames

  
class ParallelVAEWorker(WorkerBase):
    """
    A worker class that executes the VAE on a GPU.
    """
    parallel_config: ParallelConfig
    def __init__(
        self,
        parallel_config: ParallelConfig,
        rank: int,
    ) -> None:
        WorkerBase.__init__(self)
        self.parallel_config = parallel_config
        self.rank = rank
        self.vae = None

        self.if_send_to_front = (parallel_config.post_parallel_size == 0)
    
    def connect_pipeline(self):
        if self.rank == self.parallel_config.dit_parallel_size:  #  if dit_parallel_size is 0, then rank 0 is the default vae worker
            assert ray.is_initialized(), "Ray is not initialized"
            # receive data from dit2vae_latents_queue, send data to vae2post_latents_queue
            if not hasattr(self, 'recv_dit2vae_queue_manager'):
                self.recv_dit2vae_queue_manager = ray.get_actor("dit2vae_queue", namespace='matrix')
            if not hasattr(self, 'send_vae2post_queue_manager'):
                if self.if_send_to_front:
                    self.send_vae2post_queue_manager = ray.get_actor("post2front_queue", namespace='matrix')
                else:
                    self.send_vae2post_queue_manager = ray.get_actor("vae2post_queue", namespace='matrix') 
    
    def init_worker_distributed_environment(self):
        print("init_worker_distributed_environment of ParallelVAEWorker")
        print(f"Ray Rank {self.rank} is running the Parallel VAE Worker")
        # print('Arguments when init the worker: ', self.rank, self.parallel_config.world_size)
        # print('Env Variables of worker: ', os.environ)
        assert "MASTER_ADDR" in os.environ, "MASTER_ADDR is not set in the worker"
        assert "MASTER_PORT" in os.environ, "MASTER_PORT is not set in the worker"
        init_distributed_environment(  # `init_process_group` + `set_device` in each worker
            rank=self.rank,
            world_size=self.parallel_config.world_size,
        )
        # if only running the vae parallel, setting `dit_parallel_size` to 0
        # init_vae_group: _VAE = torch.distributed.new_group(ranks=vae_ranks, backend=backend) --> create a new custom group for vae workers
        # vae_ranks is hardcoded to `list(range(dit_parallel_size, dit_parallel_size + vae_parallel_size))`
        init_vae_group(self.parallel_config.dit_parallel_size, self.parallel_config.vae_parallel_size, torch.distributed.Backend.NCCL)
        VAEParallelState.initialize(vae_group=get_vae_parallel_group())
        # TODO: add init_dit_group
        
        # postprocessor group is not needed because it only has one worker
        
    def from_pretrained(
        self, 
        pretrained_model_name_or_path: str,
        **kwargs
    ):
        parallel_decoding_idx = 0
        local_rank = get_world_group().local_rank
        vae = AutoencoderKLCogVideoX.from_pretrained(os.path.join(pretrained_model_name_or_path, 'vae'),  **kwargs)
        vae.enable_parallel_decoding(parallel_decoding_idx)
        vae = vae.to(f"cuda:{local_rank}")  
        assert local_rank == 0, "Local rank is not 0"  # TODO: Is local_rank always 0 here in the ray?
        self.vae = ParallelVAEWrapper(vae) # Wrapper could handle communication with other workers using torch.dist / apply video_processor 
        return
    
    def prepare_run(self, input_config, steps: int = 3, sync_steps: int = 1):
        # This member function cannot be left unimplemented because it is defined as an abstract method in its parent class (ABC), which requires all subclasses to implement it.
        return None
    
    def post_process(self, frames_pt):
        return frames_pt_to_numpy_list(frames_pt)
    
    def execute(self, **kwargs):
        frames = self.vae.execute(**kwargs)  # this will run `execute` of xxxWrapper
        if self.if_send_to_front:
            frames = self.post_process(frames)
        return frames
    
    def _get_latents(self):
        latents = ray.get(self.recv_dit2vae_queue_manager.get.remote())
        latents, passed_time = get_data_and_passed_time(latents)
        if passed_time is not None:
            print(f"[ParallelVAEWorker._get_latents] {passed_time=}s")
        return latents
    
    def get_latents(self, **kwargs):
        # receive latents from DiT worker and send to all VAE worker
        # This method can be modified to handle commnuication with other workers
        # ref: https://github.com/xdit-project/xDiT/blob/8f4b9d30ccf278aef1e7ae985f59f7c186371d41/xfuser/model_executor/pipelines/base_pipeline.py
        # if # first vae worker
        global_rank = get_world_group().rank
        first_vae_worker_rank = self.parallel_config.dit_parallel_size
        local_rank = get_world_group().local_rank
        dtype = torch.bfloat16  # TODO: dtype could be a member variable set by engine conifg: https://github.com/xdit-project/xDiT/blob/8f4b9d30ccf278aef1e7ae985f59f7c186371d41/xfuser/model_executor/pipelines/base_pipeline.py#L102
        device = torch.device(f"cuda:{local_rank}")
        vae_parallel_size = self.parallel_config.vae_parallel_size
        if vae_parallel_size == 1:
            print("[ParallelVAEWorker.get_latents] vae_parallel_size == 1")
            latents = self._get_latents()
            dit2vae_qsize = ray.get(self.recv_dit2vae_queue_manager.qsize.remote())
            print(f"[ParallelVAEWorker.get_latents] Rank {global_rank} received the latents, {dit2vae_qsize=}")
            latents = latents.to(dtype=dtype, device=device)
            return latents
        else:
            if global_rank == first_vae_worker_rank:
                latents = self._get_latents()
                dit2vae_qsize = ray.get(self.recv_dit2vae_queue_manager.qsize.remote())
                print(f"[ParallelVAEWorker.get_latents] Rank {global_rank} received the latents, {dit2vae_qsize=}")
                latents = latents.to(dtype=dtype, device=device)  # move to GPU
                if not hasattr(self, 'shape_tensor'):
                    shape_len = torch.tensor([len(latents.shape)], dtype=torch.int, device=device)
                    shape_tensor = torch.tensor(latents.shape, dtype=torch.int, device=device)
                    
                    print(f"[ParallelVAEWorker.get_latents] shape_len: {shape_len}, shape_tensor: {shape_tensor}")
                    # Broadcast data to VAE group (makes all ranks in the vae group update their tensor to match the tensor on src rank.)
                    torch.distributed.broadcast(shape_len, src=global_rank, group=get_vae_parallel_group())  # use the default group to send the data
                    torch.distributed.broadcast(shape_tensor, src=global_rank, group=get_vae_parallel_group())
                    self.shape_tensor = shape_tensor
                torch.distributed.broadcast(latents, src=global_rank, group=get_vae_parallel_group())
                print(f"[ParallelVAEWorker.get_latents] Rank {global_rank} is broadcasting the latents")
            else:
                print(f"[ParallelVAEWorker.get_latents] Rank {global_rank} is waiting for the latents")
                # Other VAE ranks receive broadcast
                if not hasattr(self, 'shape_tensor'):
                    shape_len = torch.zeros(1, dtype=torch.int, device=device)
                    torch.distributed.broadcast(shape_len, src=first_vae_worker_rank, group=get_vae_parallel_group())
                    shape_tensor = torch.zeros(shape_len[0], dtype=torch.int, device=device)
                    torch.distributed.broadcast(shape_tensor, src=first_vae_worker_rank, group=get_vae_parallel_group())
                    self.shape_tensor = shape_tensor
                latents = torch.zeros(torch.Size(self.shape_tensor), dtype=dtype, device=device)
                torch.distributed.broadcast(latents, src=first_vae_worker_rank, group=get_vae_parallel_group())
                print(f"[ParallelVAEWorker.get_latents] Rank {global_rank} is receiving the latents")
        return latents
    
    def _send_frames(self, frames, blocking=True):
        frames = add_timestamp(frames)
        if blocking:
            ray.get(self.send_vae2post_queue_manager.put.remote(frames))
        else:
            self.send_vae2post_queue_manager.put.remote(frames)

    def send_frames(self, frames):
        if self.rank == self.parallel_config.dit_parallel_size:
            assert hasattr(self, 'send_vae2post_queue_manager'), "send_vae2post_queue_manager is not defined on the first vae worker"
            print(f"[ParallelVAEWorker.send_frames] Rank {self.rank} is sending the frames")
            if torch.is_tensor(frames):
                frames = frames.to(torch.device('cpu'))  # move the frames to CPU before sending via ray
                print("[ParallelVAEWorker.send_frames] frames shape: ", frames.shape)
                self._send_frames(frames, blocking=True)
            else:
                assert isinstance(frames, list), "frames should be a list of images in numpy array format"
                print(f"[ParallelVAEWorker.send_frames] {len(frames)=}, {frames[0].shape=}")
                for frame in frames:
                    self._send_frames(frame, blocking=False)
            print(f"[ParallelVAEWorker.send_frames] Rank {self.rank} sent the frames")
            
    def background_loop(self, **kwargs):
        self.connect_pipeline()
        latents_window = []
        while True:
            # Only the first worker will receive the latent from queue, and distribute the data to all other workers
            with timer(
                label=f"[ParallelVAEWorker.background_loop] `self.get_latents`",
                if_print=self.rank == self.parallel_config.dit_parallel_size,
            ):
                latents = self.get_latents()  # every worker will receive the same latents
            # ======= will be removed after the vae worker's cache is fixed =======
            min_num_latents = 2
            if latents.shape[1] == 1:
                latents_window.append(latents)
                if len(latents_window) < min_num_latents:
                    print(f"[ParallelVAEWorker.background_loop] Not enough latents to decode")
                    continue
                latents = torch.cat(latents_window, axis=1)
                latents_window = []
            # =====================================================================
            # All workers process the same latents and produce the same frames (communication is automatically handled in the forward within the vae group)
            with timer(
                label=f"[ParallelVAEWorker.background_loop] `self.execute`",
                if_print=self.rank == self.parallel_config.dit_parallel_size,
            ):
                frames = self.execute(latents=latents)
            # Only the first worker will send the frames to the postprocessor queue  
            with timer(
                label=f"[ParallelVAEWorker.background_loop] `self.send_frames`",
                if_print=self.rank == self.parallel_config.dit_parallel_size,
            ):
                self.send_frames(frames)


class PostProcessorWorker(WorkerBase):
    parallel_config: ParallelConfig
    def __init__(
        self,
        parallel_config: ParallelConfig,
        rank: int,
    ) -> None:
        WorkerBase.__init__(self)
        assert parallel_config.post_parallel_size == 1, "Postprocessor only support one worker"
        self.parallel_config = parallel_config
        self.rank = rank
        self.postprocessor = None
        self.last_image = None
    
    def init_worker_distributed_environment(self):
        print("init_worker_distributed_environment of PostProcessorWorker")
        # PoseProcessor only need one GPU so doesn't need to init dist environment
        print(f"Ray Rank {self.rank} is running the PostProcessor")
        # postprocessor group is not needed because it only has one worker, but by being registered in the world group, it can still communicate with other workers
        init_distributed_environment(  # `init_process_group` + `set_device` in each worker
            rank=self.rank,
            world_size=self.parallel_config.world_size,
        )
    
    def connect_pipeline(self):
        assert self.rank == self.parallel_config.dit_parallel_size + self.parallel_config.vae_parallel_size, "Postprocessor worker should be the last effective worker"
        assert ray.is_initialized(), "Ray is not initialized"
        
        if not hasattr(self, 'recv_vae2post_queue_manager'):
            # receive data from dit2vae_latents_queue
            self.recv_vae2post_queue_manager = ray.get_actor("vae2post_queue", namespace='matrix')
        if not hasattr(self, 'send_post2front_queue_manager'):
            # send data to vae2post_latents_queue
            self.send_post2front_queue_manager = ray.get_actor("post2front_queue", namespace='matrix') 
        
    def from_pretrained(
        self, 
        pretrained_model_name_or_path: str,
        **kwargs
    ): 
        print("[PostProcessorWorker.from_pretrained] Init models...")
        # init frame interpolator
        frame_interpolator_model_path = kwargs.get('frame_interpolator_model_path', None)
        self.frame_interpolator = None
        local_rank = get_world_group().local_rank
        if frame_interpolator_model_path:
            frame_interpolator = RIFEInterpolator(model_path=frame_interpolator_model_path)
            frame_interpolator.model.flownet.requires_grad_(False)
            frame_interpolator.model.flownet.eval()
            # frame_interpolator.model.flownet = torch.compile(frame_interpolator.model.flownet, mode="max-autotune-no-cudagraphs")
            frame_interpolator.model.flownet = frame_interpolator.model.flownet.to(f"cuda:{local_rank}")
            self.frame_interpolator = frame_interpolator
            
        # init video processor (convert frames tensor to PIL)
        vae_config_block_out_channels = [128, 256, 256, 512]
        vae_scale_factor_spatial = 2 ** (len(vae_config_block_out_channels) - 1)
        self.video_processor = VideoProcessor(vae_scale_factor=vae_scale_factor_spatial)
        
        return

    def frame_interpolation(self, frame_list):
        raise NotImplementedError("Interpolation for frame in `np.array` format is not implemented yet")
        # TODO: Now different section use PIL image list as intermediate result dtype, but journee needs `np.array` in np.uint8 for display
        # TODO: Implement more efficient frame interpolation using NVIDIA FRUC
        interpolated_frame_list = self.frame_interpolator.interpolate(frame_list, multi=4)
        return interpolated_frame_list
    
    def super_resolution(self, frame_list):
        # TODO: Implement super resolution
        return frame_list
    
    def postprocess(self, frame_list):
        # postprocess the images (e.g. super resolution, frame interpolation)
        if self.frame_interpolator is not None:
            frame_list = self.frame_interpolation(frame_list)
        # print("interpolated image list: ", frame_list)
        frame_list = self.super_resolution(frame_list)
        # print("super resolution image list: ", frame_list)
        return frame_list
    
    def prepare_run(self, input_config, steps: int = 3, sync_steps: int = 1):
        # This member function cannot be left unimplemented because it is defined as an abstract method in its parent class (ABC), which requires all subclasses to implement it.
        return None
    
    def execute(self, **kwargs):
        # the previous last frame need to be inserted to the beginning of the new frames to do the interpolation, and being removed after postprocessing
        frames = kwargs.get('frames', None)
        print(f"[PostProcessorWorker.execute] {frames.shape=}")
        assert frames is not None
        # pt -> numpy -> umpy list
        image_list = frames_pt_to_numpy_list(frames)
        print(f"[PostProcessorWorker.execute] {len(image_list)=}, {image_list[0].shape=}")
        if self.frame_interpolator is not None and self.last_image is not None:
            full_image_list = [self.last_image] + image_list
        else:
            full_image_list = image_list
        print("[PostProcessorWorker.execute] {len(full_image_list)=}")
        post_image_list = self.postprocess(full_image_list)
        print("[PostProcessorWorker.execute] {len(post_image_list)=}")
        if self.frame_interpolator is not None and self.last_image is not None:
            post_image_list = post_image_list[1:]
        self.last_image = post_image_list[-1]
        return post_image_list
    
    def background_loop(self, **kwargs):
        print("[PostProcessorWorker.background_loop] Running...")
        assert self.rank == self.parallel_config.dit_parallel_size + self.parallel_config.vae_parallel_size
        self.last_image = None  # reset the last image
        self.connect_pipeline()
        # This is based on only one post-processing worker, so no need to worry about synchronization
        while True:
            print("[PostProcessorWorker.background_loop] PostProcessorWorker is waiting for the frames")
            frames = ray.get(self.recv_vae2post_queue_manager.get.remote())  # this should be blocking
            print("[PostProcessorWorker.background_loop] PostProcessorWorker received the frames")
            post_image_list = self.execute(frames=frames)
            print(f"[PostProcessorWorker.background_loop] PostProcessorWorker is sending the post-processed images, {len(post_image_list)=}")
            for image in post_image_list:
                self.send_post2front_queue_manager.put.remote(image)  # this could be non-blocking