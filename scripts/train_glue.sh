#!/usr/bin/env bash
set -euo pipefail

TASK="${1:-sst2}"
MODEL_NAME="${MODEL_NAME:-FacebookAI/roberta-base}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/glue}"

python examples/train_glue.py \
  --task "${TASK}" \
  --model_name_or_path "${MODEL_NAME}" \
  --output_dir "${OUTPUT_DIR}"

