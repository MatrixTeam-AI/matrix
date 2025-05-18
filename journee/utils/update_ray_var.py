import argparse
import ray
from utils.ray_pipeline_utils import QueueManager, SharedVar, SharedReadOnlyVar, timer

def main():
    parser = argparse.ArgumentParser(description="Update the Ray variable.")
    parser.add_argument(
        "--name", 
        type=str, 
        required=True,
        help="Message to send to the logfire platform.",
    )
    parser.add_argument(
        "--name_space", 
        type=str, 
        default="matrix",
        help="Name space of the Ray variable.",
    )
    parser.add_argument(
        "--value", 
        type=str, 
        required=True,
        help="Value to set for the Ray variable.",
    )
    args = parser.parse_args()
    # 1. Connect to an existing ray cluster
    ray.init(address='auto')  
    var_pointer = ray.get_actor(parser.name, namespace=parser.name_space)
    # 2. Update the Ray variable
    ray.get(var_pointer.set.remote(parser.value)) 


if __name__ == "__main__":
    main()