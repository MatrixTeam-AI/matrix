import argparse
import time
import ray
from utils.log_utils import redirect_stdout_err_to_logger, logger
# redirect_stdout_err_to_logger(logger)
from utils.ray_pipeline_utils import QueueManager, SharedVar, SharedReadOnlyVar, timer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create Ray queues and shared variables.")
    parser.add_argument("--action_queue_maxsize", type=int, default=10, help="Max size of action queue")
    parser.add_argument("--dit2vae_queue_maxsize", type=int, default=1, help="Max size of dit2vae queue")
    args = parser.parse_args()

    ray.init(address='auto')  # connect to ray cluster
    
    action_queue = QueueManager.options(namespace='matrix', name="action_queue").remote(maxsize=args.action_queue_maxsize)  # Create a queue for accepting action commands
    dit2vae_queue = QueueManager.options(namespace='matrix', name="dit2vae_queue").remote(maxsize=args.dit2vae_queue_maxsize)  # DiT --> VAE
    vae2post_queue = QueueManager.options(namespace='matrix', name="vae2post_queue").remote()  # VAE --> Postprocessing
    post2front_queue = QueueManager.options(namespace='matrix', name="post2front_queue").remote()  # Postprocessing --> front end
    dit_step_var = SharedReadOnlyVar.options(namespace='matrix', name="dit_step_var").remote(value=0)
    vae_step_var = SharedReadOnlyVar.options(namespace='matrix', name="vae_step_var").remote(value=0)
    
    current_state_var = SharedReadOnlyVar.options(namespace='matrix', name="current_state").remote(value="STOP")
    current_prompt_var = SharedReadOnlyVar.options(namespace='matrix', name="current_prompt").remote(value=None)
    actors = ray.util.list_named_actors(all_namespaces=True)
    print(f"Actors in all namespaces: {[actor_name for actor_name in actors]}")
    
    # === IMPORTANT NOTE ===
    # These queues and shared variables are Ray actors. If the main process exits right after creating them,
    # and no other processes are using them or holding references, Ray may automatically clean them up.
    # To keep these actors alive and available for other components (e.g., workers, web server),
    # you must keep the main process running or create them inside a persistent service.
    
    try:
        print("Queues are running. Press Ctrl+C to exit.")
        while True:
            time.sleep(10)  # Keep the process alive to prevent actors from being destroyed
    except KeyboardInterrupt:
        print("Shutting down.")
