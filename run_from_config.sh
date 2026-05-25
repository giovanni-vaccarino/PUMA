#!/bin/bash
# =============================================================================
# Run the PUMA offline pipeline from a per-model config file.
#
# Usage:
#   bash run_from_config.sh <config> <base_dir> [model]
#
#   <config>    a file under configs/, e.g. configs/DS-7B.conf
#   <base_dir>  directory containing answers.json (outputs are written here too)
#   [model]     model name/path; if omitted, you must pass it via --model below
#
# Example:
#   bash run_from_config.sh configs/DS-7B.conf runs/ds7b_math500 \
#        deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
#
# The config sets the PUMA hyperparameters (RD / LB / VEE); this script maps
# them to run_puma.py CLI flags. Every variable in the config is passed through.
# =============================================================================
set -euo pipefail

CONFIG="${1:?Usage: run_from_config.sh <config> <base_dir> [model]}"
BASE_DIR="${2:?Usage: run_from_config.sh <config> <base_dir> [model]}"
MODEL="${3:-}"

# shellcheck source=/dev/null
source "$CONFIG"

ARGS=(
  --base-dir "$BASE_DIR"
  --embedding-model "$EMBEDDING_MODEL"
  --similarity-threshold "$SIMILARITY_THRESHOLD"
  --window-size "$WINDOW_SIZE"
  --consecutive-redundancy-stop "$CONSECUTIVE_REDUNDANCY_STOP"
  --consecutive-redundancy-min-step "$CONSECUTIVE_REDUNDANCY_MIN_STEP"
  --forced-stop-min-confidence "$FORCED_STOP_MIN_CONFIDENCE"
  --confidence-threshold "$CONFIDENCE_THRESHOLD"
  --epsilon "$EPSILON"
  --consecutive "$CONSECUTIVE"
)
if [ -n "$MODEL" ]; then
  ARGS+=(--model "$MODEL")
fi

echo "Running PUMA with config: $CONFIG"
python run_puma.py "${ARGS[@]}"
