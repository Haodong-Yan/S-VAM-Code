#!/bin/bash
# Evaluate VPP action model on Calvin ABC-D benchmark (1000 instruction chains).
# Usage: bash scripts/evaluate_calvin.sh
#
# Override paths via environment variables:
#   ACTION_MODEL_CKPT  - path to action model checkpoint (.pt file)
#   CALVIN_DATA_DIR    - path to calvin/dataset/task_ABC_D
#   VIDEO_MODEL_PATH   - path to svd-robot-calvin-ft checkpoint
#   CLIP_MODEL_PATH    - path to clip-vit-base-patch32

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

ACTION_MODEL_CKPT="${ACTION_MODEL_CKPT:?Please set ACTION_MODEL_CKPT to path of action model checkpoint}"
CALVIN_DATA_DIR="${CALVIN_DATA_DIR:?Please set CALVIN_DATA_DIR to path of calvin/dataset/task_ABC_D}"
VIDEO_MODEL_PATH="${VIDEO_MODEL_PATH:?Please set VIDEO_MODEL_PATH to path of svd-robot-calvin-ft}"
CLIP_MODEL_PATH="${CLIP_MODEL_PATH:?Please set CLIP_MODEL_PATH to path of clip-vit-base-patch32}"

CALVIN_ENV_PYTHON_ROOT="${SCRIPT_DIR}/calvin/calvin_env"
if [[ ! -d "${CALVIN_ENV_PYTHON_ROOT}/calvin_env" ]]; then
  echo "[WARNING] calvin_env not found at ${CALVIN_ENV_PYTHON_ROOT}, evaluation may fail." >&2
fi

export PYTHONPATH="${CALVIN_ENV_PYTHON_ROOT}:${SCRIPT_DIR}:${PYTHONPATH:-}"
cd "${SCRIPT_DIR}"

python policy_evaluation/calvin_evaluate_our.py \
  --action_model_folder "${ACTION_MODEL_CKPT}" \
  --calvin_abc_dir "${CALVIN_DATA_DIR}" \
  --video_model_path "${VIDEO_MODEL_PATH}" \
  --clip_model_path "${CLIP_MODEL_PATH}" \
  --use_ref_frame \
  --use_dpa_ref_frame \
  --use_hidden_dino_concat \
  --use_hidden_dino_dpa_concat \
  --disable_gripper_features \
  --force_eval
