import ray

class QueueInterface:
    def __init__(
        self, 
        queue_ray_actor
    ):
        self.queue_ray_actor = queue_ray_actor

    def get(self):
        item = ray.get(self.queue_ray_actor.get.remote())   # block
        return item

    def put(self, item):
        self.queue_ray_actor.put.remote(item)   # non-block

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