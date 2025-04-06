#!/bin/bash
CUDA_VISIBLE_DEVICES=2,3 ray start --include-dashboard=True --head
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cleanup() {
  echo "🧹 Cleaning up background processes..."
  kill $BACK_PID_0 $BACK_PID_1 $BACK_PID_2 $BACK_PID_3
  exit
}

trap cleanup SIGINT
cd journee
python create_ray_pipe.py \
  --action_queue_maxsize 2 &
BACK_PID_0=$!
sleep 10
# this will use the GPU designated by CUDA_VISIBLE_DEVICES to `ray start`
python start_decoding_daemon.py \
  --model_path "/MODEL/PATH" \
  --dit_parallel_size 0 \
  --vae_parallel_size 2 \
  --post_parallel_size 0 &
BACK_PID_1=$!
bash start_dit.sh &
BACK_PID_2=$!
cd ..

sleep 30
python main.py &
BACK_PID_3=$!