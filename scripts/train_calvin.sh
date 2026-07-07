#!/bin/bash
# Train VPP action model on Calvin ABC-D benchmark.
# Usage: bash scripts/train_calvin.sh
#
# Override paths via environment variables:
#   CALVIN_DATA_DIR   - path to calvin/dataset/task_ABC_D
#   VIDEO_MODEL_PATH  - path to svd-robot-calvin-ft checkpoint
#   TEXT_ENCODER_PATH  - path to clip-vit-base-patch32
#   LOG_DIR           - where to save logs and checkpoints
#   NUM_GPUS          - number of GPUs (default: 4)

set -euo pipefail

CALVIN_DATA_DIR="${CALVIN_DATA_DIR:?Please set CALVIN_DATA_DIR to path of calvin/dataset/task_ABC_D}"
VIDEO_MODEL_PATH="${VIDEO_MODEL_PATH:?Please set VIDEO_MODEL_PATH to path of svd-robot-calvin-ft}"
TEXT_ENCODER_PATH="${TEXT_ENCODER_PATH:?Please set TEXT_ENCODER_PATH to path of clip-vit-base-patch32}"
LOG_DIR="${LOG_DIR:-./logs/s_vam_calvin}"
NUM_GPUS="${NUM_GPUS:-4}"

accelerate launch \
  --num_processes "${NUM_GPUS}" \
  --num_machines 1 \
  step2_train_action_calvin.py \
  --root_data_dir "${CALVIN_DATA_DIR}" \
  --video_model_path "${VIDEO_MODEL_PATH}" \
  --text_encoder_path "${TEXT_ENCODER_PATH}" \
  --log_dir "${LOG_DIR}" \
  --use_ref_frame \
  --use_dpa_ref_frame \
  --use_hidden_dino_concat \
  --use_hidden_dino_dpa_concat \
  --disable_gripper_features
