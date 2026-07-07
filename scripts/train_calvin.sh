#!/bin/bash
# Train VPP action model on Calvin ABC-D benchmark.
# Usage: bash scripts/train_calvin.sh
#
# Override paths via environment variables:
#   CALVIN_DATA_DIR   - path to calvin/dataset/task_ABC_D
#   VIDEO_MODEL_PATH  - path to svd-robot-calvin-ft checkpoint
#   TEXT_ENCODER_PATH  - path to clip-vit-base-patch32
#   HIDDEN2DINO_CKPT  - path to hidden2dino checkpoint
#   HIDDEN2DPA_CKPT   - path to hidden2dpa checkpoint
#   DINOV2_PATH       - path to DINOv2 torch hub directory
#   DA3_PATH          - path to Depth Anything 3 model directory
#   LOG_DIR           - where to save logs and checkpoints
#   NUM_GPUS          - number of GPUs (default: 4)

set -euo pipefail

CALVIN_DATA_DIR="${CALVIN_DATA_DIR:?Please set CALVIN_DATA_DIR to path of calvin/dataset/task_ABC_D}"
VIDEO_MODEL_PATH="${VIDEO_MODEL_PATH:?Please set VIDEO_MODEL_PATH to path of svd-robot-calvin-ft}"
TEXT_ENCODER_PATH="${TEXT_ENCODER_PATH:?Please set TEXT_ENCODER_PATH to path of clip-vit-base-patch32}"
HIDDEN2DINO_CKPT="${HIDDEN2DINO_CKPT:?Please set HIDDEN2DINO_CKPT to path of hidden2dino checkpoint}"
HIDDEN2DPA_CKPT="${HIDDEN2DPA_CKPT:?Please set HIDDEN2DPA_CKPT to path of hidden2dpa checkpoint}"
DINOV2_PATH="${DINOV2_PATH:?Please set DINOV2_PATH to path of DINOv2 torch hub directory}"
DA3_PATH="${DA3_PATH:?Please set DA3_PATH to path of Depth Anything 3 model directory}"
LOG_DIR="${LOG_DIR:-./logs/s_vam_calvin}"
NUM_GPUS="${NUM_GPUS:-4}"

accelerate launch \
  --num_processes "${NUM_GPUS}" \
  --num_machines 1 \
  step2_train_action_calvin.py \
  --root_data_dir "${CALVIN_DATA_DIR}" \
  --video_model_path "${VIDEO_MODEL_PATH}" \
  --text_encoder_path "${TEXT_ENCODER_PATH}" \
  --hidden2dino_ckpt "${HIDDEN2DINO_CKPT}" \
  --hidden2dpa_ckpt "${HIDDEN2DPA_CKPT}" \
  --dinov2_path "${DINOV2_PATH}" \
  --da3_path "${DA3_PATH}" \
  --log_dir "${LOG_DIR}"
