#!/usr/bin/env bash
set -euo pipefail

TASK="${1:-sst2}"
ADAPTER_PATH="${ADAPTER_PATH:-outputs}"
ADAPTER_SUBFOLDER="${ADAPTER_SUBFOLDER:-${TASK}/best_model}"
MODEL_NAME="${MODEL_NAME:-FacebookAI/roberta-base}"

python examples/test_glue.py \
  --task "${TASK}" \
  --model_name_or_path "${MODEL_NAME}" \
  --adapter_path "${ADAPTER_PATH}" \
  --adapter_subfolder "${ADAPTER_SUBFOLDER}"

