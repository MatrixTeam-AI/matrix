#!/bin/bash
# This file is modified from https://github.com/xdit-project/xDiT/blob/0.4.1/examples/run_cogvideo.sh
DEFAULT_NUM_GPUS=1
NUM_GPUS=${1:-$DEFAULT_NUM_GPUS}
MODEL_PATH=$2

# generate string "0,1,...,NUM_GPUS-1"
GPU_IDS=0
for ((i=1; i<NUM_GPUS; i++)); do
  GPU_IDS="$GPU_IDS,$i"
done

python utils/send_msg_to_logger.py --message "GPU_IDS: $GPU_IDS"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set -x

export PYTHONPATH=$PWD:$PYTHONPATH
echo $PYTHONPATH
# export HF_HOME="/mnt/world_model/longxiang/.cache/huggingface"

# export NCCL_BUFFSIZE=1048576  # for 24 GB memory, the default 32MB NCCL buffer per channel would be too large and cause OOM (These memory wouldn't be released by `torch.cuda.empty_cache()`)
CUDA_VISIBLE_DEVICES=$GPU_IDS torchrun --nnodes 1 --nproc-per-node $NUM_GPUS --master-port 29501 ./stage4_ray/inference_ulysses_interactive.py \
--model_path "$MODEL_PATH" \
--image_or_video_path ../base_video.mp4 \
--video_cache_dir ../base_video_cache \
--num_sample_groups 1000000 \
--warmup_steps 0 \
--ulysses_degree $NUM_GPUS \

# warmup_steps for xdit, default to 1
# ulysses_degree for xdit
