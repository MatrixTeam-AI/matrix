import time
from contextlib import contextmanager
import asyncio

import torch
import torch.distributed as dist
import ray
from ray.util.queue import Queue

@contextmanager
def timer(label="Block", if_print=True, print_rank=0):
    start_time = time.perf_counter()
    yield
    if torch.cuda.is_available():
        torch.cuda.synchronize()  # Ensures all CUDA operations are completed before measuring time
    end_time = time.perf_counter()
    
    if if_print and (not dist.is_initialized() or dist.get_rank() == print_rank):
        print(f"{label} took {end_time - start_time:.6f} seconds")

ADD_TIMESTAMP = True

def is_data_with_timestamps(data):
    return isinstance(data, dict) and "timestamps" in data

def add_timestamp(
    data=None,
    timestamps=None,
    update=True,
    label=None,
    sync_cuda=False,
    return_tuple=False,
    current_time=None,
):
    if not ADD_TIMESTAMP:
        if return_tuple:
            return data, None
        else:
            return data
        
    if current_time is None:
        if sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
        current_time = time.perf_counter()
    timestamp = (label, current_time)
    if not is_data_with_timestamps(data):
        data = dict(data=data, timestamps=[])
        if timestamps is not None:
            data["timestamps"] = timestamps
    if update:
        if len(data['timestamps']) > 0 and isinstance(data['timestamps'][0], list):
            # a batch of timestamps: List[List[timestamp tuple[str, float]]]
            for timestamps in data['timestamps']:
                timestamps.append(timestamp)
        else:
            # a single timestamp: List[timestamp tuple[str, float]]
            data['timestamps'].append(timestamp)
    if return_tuple:
        return data['data'], data['timestamps']
    else:
        return data
    
def add_timestamp_to_each_item(
    data_list: list,
    timestamps=None,
    update=True,
    label=None,
    sync_cuda=False,
    return_tuple=False,
):
    assert isinstance(data_list, list)
    if not ADD_TIMESTAMP:
        return data_list
    
    if sync_cuda and torch.cuda.is_available():
        torch.cuda.synchronize()
    current_time = time.perf_counter()

    return [
        add_timestamp(
            data=data,
            timestamps=timestamps,
            update=update,
            label=label,
            return_tuple=return_tuple,
            current_time=current_time,
        )
        for data in data_list
    ]

def get_data_and_timestamps(data):
    if not is_data_with_timestamps(data):
        return data, None
    return data['data'], data['timestamps']

def get_passed_times(timestamps):
    passed_times = dict()
    if timestamps is None:
        return passed_times
    for i, (label, current_time) in enumerate(timestamps[:-1]):
        label_next, time_next = timestamps[i + 1]
        label_interval = f"{label} to {label_next}"
        time_interval = time_next - current_time
        passed_times[label_interval] = time_interval
    return passed_times

def passed_times_dict_to_str(passed_times):
    passed_times_str = "\n".join([f" [{k}]: {v:.3f}s" for k, v in passed_times.items()])
    return passed_times_str

@ray.remote(num_cpus=1, max_concurrency=3)
class QueueManager:
    def __init__(self, maxsize=0):
        print("QueueManager initialized!")
        self.queue = Queue(maxsize=maxsize)

    def put(self, item):
        print(f"[QueueManager] put() called")
        self.queue.put(item)

    def get(self):
        print(f"[QueueManager] get() called")
        return self.queue.get()

    def get_all(self):
        print(f"[QueueManager] get_all() called")
        size = self.queue.size()
        return self.queue.get_nowait_batch(size)

    def print_queue(self):
        print(f"[QueueManager] Current queue: {self.queue}")

    def full(self):
        return self.queue.full()
    
    def empty(self):
        return self.queue.empty()
    
    def size(self):
        return self.queue.size()


@ray.remote
class SharedVar:
    def __init__(self, value=None):
        self.value = value
        self._condition = asyncio.Condition()

    async def set(self, new_value):
        async with self._condition:
            self.value = new_value
            self._condition.notify_all()  # Notify all waiting `.get()`

    async def get(self):
        async with self._condition:
            while self.value is None:
                await self._condition.wait()
            return self.value

# @ray.remote
# class SharedVar:
#     def __init__(self, value=None):
#         self.value = value

#     def set(self, new_value):
#         self.value = new_value

#     def get(self):
#         return self.value
           
# Usage:
# 1. Connect to an existing ray cluster
# ray.init(address='auto')  
# 2. Create an Actor
# queue_pointer = QueueManager.options(namespace='matrix', name="latent_queue").remote()
# 3. (For consumer) 
# queue_pointer = ray.get_actor("latent_queue", namespace="matrix")  # Ensure the actor is started
# content = ray.get(queue.get.remote())  # this will block until a content is available
# 4. (For producer)
# queue = ray.get_actor("latent_queue", namespace="matrix")
# ray.get(queue.put.remote(content))  # this is non-blocking