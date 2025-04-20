import time

import ray
from .utils.ray_pipeline_utils import add_timestamp, get_data_and_timestamps, get_passed_times, passed_times_dict_to_str

class QueueInterface:
    def __init__(
        self, 
        queue_ray_actor
    ):
        self.queue_ray_actor = queue_ray_actor

    def get(self):
        item = ray.get(self.queue_ray_actor.get.remote())   # blocking
        item, timestamps = add_timestamp(item, label='Display-frame', return_tuple=True)
        passed_times = get_passed_times(timestamps)
        return item, passed_times
    
    def put(self, item, blocking=False):
        item = add_timestamp(item, label='Display-control')
        if blocking:
            ray.get(self.queue_ray_actor.put.remote(item))   # blocking
        else:
            self.queue_ray_actor.put.remote(item)   # non-blocking

    def full(self):
        return ray.get(self.queue_ray_actor.full.remote())
    
    def empty(self):
        return ray.get(self.queue_ray_actor.empty.remote())
    
    def size(self):
        return ray.get(self.queue_ray_actor.size.remote())

class DummyModel:
    def start(self):
        r"""Start video generation.
        By default, if no control is given, the signal will be set to 'D'.
        """
        print(f"Dummy start() called")

    def init_generation(self):
        r"""prepare the warm-up video"""
        print(f"Dummy init_generation() called")

def init_model():
    post2front_ray_actor = ray.get_actor("post2front_queue", namespace='matrix')
    action_ray_actor = ray.get_actor("action_queue", namespace='matrix')
    return (
        DummyModel(),
        QueueInterface(post2front_ray_actor), 
        QueueInterface(action_ray_actor)
    )