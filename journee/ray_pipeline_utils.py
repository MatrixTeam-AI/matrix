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

def add_timestamp(data):
    if not ADD_TIMESTAMP:
        return data
    start_time = time.perf_counter()
    return dict(
        data = data,
        timestamp = start_time,
    )

def get_data_and_passed_time(data):
    if not (isinstance(data, dict) and "timestamp" in data):
        return data, None
    end_time = time.perf_counter()
    passed_time = end_time - data["timestamp"]
    return data["data"], passed_time

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

    def get_batch(self):
        """一次性把队列里当前已有的所有元素全拿出来，返回列表。"""
        items = []
        # ray.util.queue.Queue 没有原生的 get_nowait() 方法，但如果 queue.empty() == False，
        # 那么 self.queue.get() 会立刻返回，而不会阻塞。
        while not self.queue.empty():
            items.append(self.queue.get())
        return items

    def print_queue(self):
        print(f"[QueueManager] Current queue: {self.queue}")

    def full(self):
        return self.queue.full()
    
    def empty(self):
        return self.queue.empty()
    
    def qsize(self):
        return self.queue.qsize()


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