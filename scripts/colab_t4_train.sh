#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-0.5B-Instruct}"
EPISODES="${EPISODES:-12}"
GROUP_SIZE="${GROUP_SIZE:-3}"
LEARNING_RATE="${LEARNING_RATE:-5e-6}"
TEMPERATURE="${TEMPERATURE:-0.9}"
SEED="${SEED:-7}"
SAVE_PATH="${SAVE_PATH:-/content/grpo_trained_model}"
USE_LORA="${USE_LORA:-1}"
PUSH_TO_HUB_REPO="${PUSH_TO_HUB_REPO:-}"

echo "Installing training dependencies..."
python -m pip install --upgrade pip
python -m pip install -e ".[train]"

echo "Checking CUDA runtime..."
python - <<'PY'
import torch
print({"cuda_available": torch.cuda.is_available(), "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None})
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available. In Colab choose Runtime > Change runtime type > T4 GPU, then reconnect.")
PY

TRAIN_ARGS=(
  --mode manual
  --model "$MODEL_NAME"
  --episodes "$EPISODES"
  --group-size "$GROUP_SIZE"
  --learning-rate "$LEARNING_RATE"
  --temperature "$TEMPERATURE"
  --seed "$SEED"
  --device cuda
  --save-path "$SAVE_PATH"
)

if [[ "$USE_LORA" == "1" ]]; then
  TRAIN_ARGS+=(--lora)
fi

if [[ -n "$PUSH_TO_HUB_REPO" ]]; then
  TRAIN_ARGS+=(--push-to-hub-repo "$PUSH_TO_HUB_REPO")
fi

echo "Starting GRPO training on Colab T4..."
python train_grpo.py "${TRAIN_ARGS[@]}"
