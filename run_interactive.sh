#!/bin/bash
cleanup() {
   echo "🧹 Cleaning up ray processes..."
   ray stop
   exit 0
}

trap cleanup SIGINT

ray stop
python journee/utils/send_msg_to_logger.py --message "Start running run_journee.sh"

NUM_GPUS_DIT=1
NUM_GPUS_VAE=3
MODEL_PATH="../models/stage4"

generate string "NUM_GPUS_DIT,...,NUM_GPUS_DIT+NUM_GPUS_VAE-1"
GPU_IDS=$NUM_GPUS_DIT
for ((i=NUM_GPUS_DIT+1; i<NUM_GPUS_DIT+NUM_GPUS_VAE; i++)); do
  GPU_IDS="$GPU_IDS,$i"
done

python journee/utils/send_msg_to_logger.py --message "GPU_IDS: $GPU_IDS"

# download model ckpts if needed
python journee/utils/send_msg_to_logger.py --message "Download model weights..."
# bash download_hf_models.sh
# bash download_models.sh

CUDA_VISIBLE_DEVICES=$GPU_IDS ray start --head
sleep 10
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# cleanup() {
#   echo "🧹 Cleaning up background processes..."
#   kill $BACK_PID_0 $BACK_PID_1 $BACK_PID_2 $BACK_PID_3
#   exit
# }
# trap cleanup SIGINT

cd journee
python utils/send_msg_to_logger.py --message "Running python create_ray_pipe.py..."
python create_ray_pipe.py &
# python create_ray_pipe.py > ../create_ray_pipeline_output.log 2>&1 &  # for debugging
# BACK_PID_0=$!
python utils/send_msg_to_logger.py --message "Complete python create_ray_pipe.py"
sleep 30
cd ..

python journee/utils/send_msg_to_logger.py --message "Running python main.py"
python main.py &
# python main.py > main_output.log 2>&1 &  # for debugging
# BACK_PID_3=$!

cd journee
python utils/send_msg_to_logger.py --message "Running bash start_dit.sh"
bash start_dit.sh $NUM_GPUS_DIT $MODEL_PATH &
# BACK_PID_2=$!

# this will use the GPU designated by CUDA_VISIBLE_DEVICES to `ray start`
python utils/send_msg_to_logger.py --message "Running python start_decoding_daemon.py"
python start_decoding_daemon.py \
  --model_path "$MODEL_PATH" \
  --dit_parallel_size 0 \
  --vae_parallel_size $NUM_GPUS_VAE \
  --post_parallel_size 0
# BACK_PID_1=$!