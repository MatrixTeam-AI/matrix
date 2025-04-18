#!/bin/bash
NUM_GPUS_DIT=3
NUM_GPUS_VAE=5

# generate string "NUM_GPUS_DIT,...,NUM_GPUS_DIT+NUM_GPUS_VAE-1"
GPU_IDS=$NUM_GPUS_DIT
for ((i=NUM_GPUS_DIT+1; i<NUM_GPUS_DIT+NUM_GPUS_VAE; i++)); do
  GPU_IDS="$GPU_IDS,$i"
done

# download model ckpts if needed
bash download_hf_models.sh

CUDA_VISIBLE_DEVICES=1,2,3 ray start --head
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cleanup() {
  echo "🧹 Cleaning up background processes..."
  kill $BACK_PID_0 $BACK_PID_1 $BACK_PID_2 $BACK_PID_3
  exit
}

trap cleanup SIGINT
cd journee
python create_ray_pipe.py \
  --action_queue_maxsize 100 &
BACK_PID_0=$!
sleep 10
# this will use the GPU designated by CUDA_VISIBLE_DEVICES to `ray start`
python start_decoding_daemon.py \
  --model_path "../models/stage3" \
  --dit_parallel_size 0 \
  --vae_parallel_size $NUM_GPUS_VAE \
  --post_parallel_size 0 &
BACK_PID_1=$!
bash start_dit.sh $NUM_GPUS_DIT &
BACK_PID_2=$!
cd ..

sleep 30
python main.py &
BACK_PID_3=$!