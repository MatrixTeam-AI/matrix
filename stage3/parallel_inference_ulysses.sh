#!/bin/bash
# This file is modified from https://github.com/xdit-project/xDiT/blob/0.4.1/examples/run_cogvideo.sh
set -x

export PYTHONPATH=$PWD:$PYTHONPATH

# Output folder
OUTDIR="./output/ulysses"
mkdir -p $OUTDIR

# CogVideoX configuration
SCRIPT="inference_ulysses.py"

# CogVideoX specific task args
TASK_ARGS="--height 480 --width 720" # 480 : 720 = 3 : 4
CFG_SCALE=1
SEED=43

# CogVideoX parallel configuration
N_GPUS=3
PARALLEL_ARGS="--ulysses_degree 3 --ring_degree 1"
# CFG_ARGS="--use_cfg_parallel"
SPLIT_TEXT_EMBED_IN_SP="true"

# export NCCL_MIN_NCHANNELS
export NCCL_DEBUG=VERSION
# unset NCCL_NSOCKS_PERTHREAD
export NCCL_MAX_NCHANNELS=64
# unset NCCL_ASYNC_ERROR_HANDLING
# unset NCCL_SOCKET_NTHREADS
# unset NCCL_LAUNCH_MODE



# Uncomment and modify these as needed
# PIPEFUSION_ARGS="--num_pipeline_patch 8"
# OUTPUT_ARGS="--output_type latent"
PARALLLEL_VAE="--parallel_decoding_idx 0"
# ENABLE_TILING="--enable_tiling"
# COMPILE_FLAG="--use_torch_compile"

torchrun --nproc_per_node=$N_GPUS ./$SCRIPT \
--model_path MODEL_PATH \
--output_path "${OUTDIR}/output_seed${SEED}_cfgscale${CFG_SCALE}_splitText-${SPLIT_TEXT_EMBED_IN_SP}.mp4" \
--prompt "The video shows a white car driving on a country road on a sunny day. The car comes from the back of the scene, moving forward along the road, with open fields and distant hills surrounding it. As the car moves, the vegetation on both sides of the road and distant buildings can be seen. The entire video records the car's journey through the natural environment using a follow-shot technique." \
--image_or_video_path VIDEO_PATH \
--control_signal D,D,D,D,DL,DL,DL,DL,DL,DL,D,D,D,D,D,D,D \
--guidance_scale $CFG_SCALE \
--seed $SEED \
--split_text_embed_in_sp $SPLIT_TEXT_EMBED_IN_SP \
--num_sample_groups 100 \
--control_repeat_length 5 \
$PARALLEL_ARGS \
$TASK_ARGS \
$PIPEFUSION_ARGS \
$OUTPUT_ARGS \
--warmup_steps 0 \
$CFG_ARGS \
$PARALLLEL_VAE \
$ENABLE_TILING \
$COMPILE_FLAG