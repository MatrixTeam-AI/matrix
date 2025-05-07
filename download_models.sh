#!/bin/bash

LOCAL_DIR="models"

REPO_ID="MatrixTeam/TheMatrix"
python journee/utils/send_msg_to_logger.py --message "Download stage3 model weights from HuggingFace..."
huggingface-cli download "$REPO_ID" --local-dir "$LOCAL_DIR" --include="stage3/*"

python journee/utils/send_msg_to_logger.py --message "Download stage4 model weights from HuggingFace..."
huggingface-cli download "$REPO_ID" --local-dir "$LOCAL_DIR" --include="stage4/*"

# REPO_ID="ztyang196/TheMatrix"
# python journee/utils/send_msg_to_logger.py --message "Download model weights from ModelScope..."
# # modelscope download "$REPO_ID" --local_dir "$LOCAL_DIR" --include="stage4/*"
# python download_ms_models.py
# python journee/utils/send_msg_to_logger.py --message "Complete downloading."

rm -rf models/stage3/transformer/*
cp models/stage4/* models/stage3/transformer/