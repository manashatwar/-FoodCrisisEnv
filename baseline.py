# HF_TOKEN is intentionally blank — use local model inference only (no API calls)
HF_TOKEN=""
API_BASE_URL=""
MODEL_NAME="Qwen/Qwen2.5-0.5B-Instruct"
LORA_PATH=""

# Local inference mode (default):
#   - HF_TOKEN must be empty
#   - API_BASE_URL must be empty
#   - Model is downloaded from HuggingFace Hub on first run and cached locally
#   - Runs entirely on local GPU/CPU via transformers
#
# To override token cap for local generation:
#   LOCAL_MAX_NEW_TOKENS=32
