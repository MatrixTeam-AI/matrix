import os
import sys
from dataclasses import dataclass
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Tuple
from itertools import islice, repeat

import torch
from diffusers.utils import export_to_video, load_image, load_video
from diffusers.video_processor import VideoProcessor

import ray
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from xfuser.ray.pipeline.ray_utils import initialize_ray_cluster
from xfuser.ray.pipeline.pipeline_utils import GPUExecutor
from xfuser.ray.worker.worker_wrappers import RayWorkerWrapper

from matrix_ray_worker import EngineConfig, ParallelConfig
from utils.ray_pipeline_utils import timer
from utils.log_utils import logger_info as print

class RayMatrixPipeline(GPUExecutor):
    runtime_env = {}
    postprocessor_workers = []
    vae_workers = []
    dit_workers = []
    
    def _init_executor(self):
        self._init_ray_workers()
        print(f"[RayMatrixPipeline._init_executor] {self.workers=}")
        self._run_workers(self.workers,"init_worker_distributed_environment")

    def _init_ray_workers(self):
        # Assume we have `world_size` gpus, and we want to split them into dit_parallel_size, vae_parallel_size, post_parallel_size for DiT, VAE, and postprocessor workers
        # dit_workers rank: [0, dit_parallel_size)
        # vae_workers rank: [dit_parallel_size, dit_parallel_size + vae_parallel_size)
        # postprocessor_workers rank: [dit_parallel_size + vae_parallel_size, world_size)
        world_size, post_parallel_size, vae_parallel_size, dit_parallel_size = self.engine_config.parallel_config.world_size, \
            self.engine_config.parallel_config.post_parallel_size, self.engine_config.parallel_config.vae_parallel_size, self.engine_config.parallel_config.dit_parallel_size
        assert world_size == dit_parallel_size + vae_parallel_size + post_parallel_size, "world_size should be equal to post_parallel_size + vae_parallel_size + dit_parallel_size"
        assert dit_parallel_size == 0, "Not implemented yet for dit workers"
        
        # connect to ray cluster and create a placement group with bundle: [{device_str: 1.0} for _ in range(parallel_config.world_size)]
        placement_group = initialize_ray_cluster(self.engine_config.parallel_config, 'auto') 
        # print("placement_group: ", placement_group) 
        # create placement group and worker wrapper instance for lazy load worker
        self.workers = []
        for bundle_id, bundle in enumerate(placement_group.bundle_specs):
            # print("bundle_id: ", bundle_id)
            # print("bundle: ", bundle)
            # Skip bundles without GPUs
            if not bundle.get("GPU", 0):
                continue

            scheduling_strategy = PlacementGroupSchedulingStrategy(
                placement_group=placement_group,
                placement_group_bundle_index=bundle_id,
                placement_group_capture_child_tasks=True,
            )
            
            if bundle_id < dit_parallel_size:
                print(f"bundle_id: {bundle_id} is DiT worker")
                #TODO: Include DiT workers support later
                raise NotImplementedError("Not implemented yet for dit workers")
            elif bundle_id < dit_parallel_size + vae_parallel_size:
                print(f"bundle_id: {bundle_id} is VAE worker")
                # the rest of them are VAE workers
                worker = ray.remote(
                    num_cpus=0,
                    num_gpus=1,
                    scheduling_strategy=scheduling_strategy,
                    runtime_env=self.runtime_env,
                )(RayWorkerWrapper).remote(
                    self.engine_config.parallel_config,
                    "matrix_ray_worker.ParallelVAEWorker",
                    bundle_id,  # GPU ID (GPU managed by ray cluster, starting from 0, but may not be the same as the GPU ID in the host machine)
                )
                self.vae_workers.append(worker)
            elif bundle_id < dit_parallel_size + vae_parallel_size + post_parallel_size:
                print(f"bundle_id: {bundle_id} is Postprocessor worker")
                worker = ray.remote(
                    num_cpus=0,
                    num_gpus=1,
                    scheduling_strategy=scheduling_strategy,
                    runtime_env=self.runtime_env,
                )(RayWorkerWrapper).remote(
                    self.engine_config.parallel_config,
                    "matrix_ray_worker.PostProcessorWorker",
                    bundle_id,  # GPU ID (GPU managed by ray cluster, starting from 0, but may not be the same as the GPU ID in the host machine)
                )
                self.postprocessor_workers.append(worker)
            
            self.workers.append(worker)

    def _run_workers(
        self,
        workers: List[ray.ObjectRef],
        method: str,
        *args,
        async_run_tensor_parallel_workers_only: bool = False,
        all_args: Optional[List[Tuple[Any, ...]]] = None,
        all_kwargs: Optional[List[Dict[str, Any]]] = None,
        **kwargs,
    ) -> Any:
        """Runs the given method on all workers. Can be used in the following
        ways:

        Args:
        - async_run_tensor_parallel_workers_only: If True the method will be
          run only in the remote TP workers, not the driver worker.
          It will also be run asynchronously and return a list of futures
          rather than blocking on the results.
        - args/kwargs: All workers share the same args/kwargs
        - all_args/all_kwargs: args/kwargs for each worker are specified
          individually
        """
        is_non_blocking_worker = kwargs.get("is_non_blocking_worker", False)
        count = len(workers)
        # If using SPMD worker, all workers are the same, so we should execute
        # the args on all workers. Otherwise, we skip the first worker's args
        # because those args will go to the driver worker.
        first_worker_args_index: int = 0
        all_worker_args = repeat(args, count) if all_args is None \
            else islice(all_args, first_worker_args_index, None)
        all_worker_kwargs = repeat(kwargs, count) if all_kwargs is None \
            else islice(all_kwargs, first_worker_args_index, None)
        # print("method: ", method)
        # print("all_worker_args: ", all_worker_args)
        # print("all_worker_kwargs: ", all_worker_kwargs)
        # Start the ray workers first.
        ray_workers = workers
        ray_worker_outputs = [
            # `execute_method` is defined in `RayWorkerWrapper`, will call the `method` of the worker
            worker.execute_method.remote(method, *worker_args, **worker_kwargs)  
            for (worker, worker_args, worker_kwargs
                 ) in zip(ray_workers, all_worker_args, all_worker_kwargs)
        ]

        if async_run_tensor_parallel_workers_only:
            # Just return futures
            return ray_worker_outputs

        # Get the results of the ray workers.
        if self.workers and not is_non_blocking_worker:
            ray_worker_outputs = ray.get(ray_worker_outputs)

        return ray_worker_outputs
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, engine_config: EngineConfig, runtime_env, **kwargs):
        cls.runtime_env = runtime_env
        pipeline = cls(engine_config)
        # all worker will call the `from_pretrained` method of the worker to load the pretrained model and init necessary process group
        pipeline._run_workers(pipeline.workers, "from_pretrained", pretrained_model_name_or_path, **kwargs)
        return pipeline

    def start_postprocessor_loop(self, **kwargs):
        self._run_workers(self.postprocessor_workers, "background_loop", is_non_blocking_worker=True, **kwargs)
    
    def start_vae_loop(self, **kwargs):
        self._run_workers(self.vae_workers, "background_loop", is_non_blocking_worker=True, **kwargs)
        
    def call_dit(self, **kwargs):
        # TODO: Include DiT workers support later
        return self._run_workers(self.dit_workers,"execute", **kwargs)
    
    def call_vae(self, **kwargs):
        return self._run_workers(self.vae_workers,"execute", **kwargs)
    
    def call_postprocessor(self, **kwargs):
        return self._run_workers(self.postprocessor_workers,"execute", **kwargs)
    
    def connect_pipeline(self, **kwargs):
        return self._run_workers(self.vae_workers,"connect_pipeline", **kwargs)

   
def test_vae_and_post_processor():
    matrix_ckpt_path = "/matrix_ckpts/stage3/vae"
    video_output_dir = "/workspace/matrix/ray_pipeline"
    frame_interpolator_model_path = "/workspace/matrix/Practical_RIFE/train_log"
    
    latents = torch.load("/workspace/matrix/latents_100.pt")
    num_latents = 2
    # latents = latents[:, :2]  # decode 100 latents at a time will cause OOM on 4090
    
    # you can add env vars for all workers by setting this `env_var` when initializing the pipeline
    dist_env_var = {'env_vars': {"MASTER_ADDR": "localhost", "MASTER_PORT": "12355"}}  
    
    parallel_config = ParallelConfig(world_size=2, dit_parallel_size=0, vae_parallel_size=1, post_parallel_size=1)
    engine_config = EngineConfig(parallel_config=parallel_config)
    matrix_ray_pipline = RayMatrixPipeline.from_pretrained(matrix_ckpt_path, engine_config, dist_env_var, 
                                                           frame_interpolator_model_path=frame_interpolator_model_path,
                                                           torch_dtype=torch.bfloat16)
    for i in range(10):
        with timer(f"Decoding {latents.size(1)} latents by VAE"):
            frames = matrix_ray_pipline.call_vae(latents=latents[:, i:i+num_latents])[0]  # only the rank 0 worker will return the results
        frames = frames[:, :, 4:]
        print(f"{frames.shape=}")
        if parallel_config.post_parallel_size > 0:  # if there is postprocessor workers
            full_video = matrix_ray_pipline.call_postprocessor(frames=frames)
            full_video = full_video[0]  # the return of the ray workers is a list. There is only one postprocessor so we get the first element (also the only element)
        else:
            vae_config_block_out_channels = [128, 256, 256, 512]
            vae_scale_factor_spatial = 2 ** (len(vae_config_block_out_channels) - 1)
            video_processor = VideoProcessor(vae_scale_factor=vae_scale_factor_spatial)
            with timer(f"Postprocessing {latents.size(1)} latents"):
                full_video = video_processor.postprocess_video(video=frames, output_type='pil')
        video_output_path = os.path.join(video_output_dir, f"video_{i}.mp4")
        print(f"Exporting video to: {video_output_path}")
        export_to_video(full_video, video_output_path, fps=16)

def test_dit():
    dist_env_var = {'env_vars': {"MASTER_ADDR": "localhost", "MASTER_PORT": "12355"}}
    
    parallel_config = ParallelConfig(world_size=5, dit_parallel_size=2, vae_parallel_size=2, post_parallel_size=1)
    engine_config = EngineConfig(parallel_config=parallel_config)
    matrix_ray_pipline = RayMatrixPipeline(engine_config, dist_env_var)
    matrix_ray_pipline.call_dit(prompt="", max_length=100)
    
if __name__ == "__main__":
    test_vae_and_post_processor()