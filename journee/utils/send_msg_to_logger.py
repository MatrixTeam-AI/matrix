import argparse
from log_utils import logger

def main():
    parser = argparse.ArgumentParser(description="Run the WMK sample.")
    parser.add_argument(
        "--message", 
        type=str, 
        required=True,
        help="Message to send to the logfire platform.",
    )
    args = parser.parse_args()

    logger.info(f"[MESSAGE] {args.message}")

if __name__ == "__main__":
    main()