#!/bin/bash

REPO_ID="MatrixTeam/TheMatrix"
FOLDER_NAME="stage4"
LOCAL_DIR="models"

huggingface-cli download "$REPO_ID" --local-dir "$LOCAL_DIR" --include="$FOLDER_NAME/*"

if [ $? -eq 0 ]; then
    echo "Successfully download $FOLDER_NAME to $LOCAL_DIR"
else
    echo "Errors occurred during download."
fi