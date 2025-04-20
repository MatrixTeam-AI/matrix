#!/bin/bash
# This file is modified from https://github.com/xdit-project/xDiT/blob/0.4.1/examples/run_cogvideo.sh
DEFAULT_NUM_GPUS=1
NUM_GPUS=${1:-$DEFAULT_NUM_GPUS}

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

export NCCL_BUFFSIZE=1048576  # for 24 GB memory, the default 32MB NCCL buffer per channel would be too large and cause OOM (These memory wouldn't be released by `torch.cuda.empty_cache()`)
CUDA_VISIBLE_DEVICES=$GPU_IDS; torchrun --nnodes 1 --nproc-per-node $NUM_GPUS --master-port 29501 ./stage4_ray/inference_ulysses_interactive.py \
--model_path "../models/stage3" \
--output_path ../samples/journee/dit_debug/output.mp4 \
--prompt "The video shows a white car driving on a country road on a sunny day. The car comes from the back of the scene, moving forward along the road, with open fields and distant hills surrounding it. As the car moves, the vegetation on both sides of the road and distant buildings can be seen. The entire video records the car's journey through the natural environment using a follow-shot technique." \
--image_or_video_path ../base_video.mp4 \
--guidance_scale 1 \
--seed 43 \
--split_text_embed_in_sp true \
--num_sample_groups 1000000 \
--ulysses_degree $NUM_GPUS \
--height 480 \
--width 720 \
--warmup_steps 0 \
--init_video_clip_frame 17