#!/usr/bin/env bash
set -euo pipefail

BASE_MODEL="${BASE_MODEL:-baffo32/decapoda-research-llama-7B-hf}"
DATA_PATH="${DATA_PATH:-yahma/alpaca-cleaned}"

python examples/train_llama.py \
  --model_source hf \
  --model_name_or_path "${BASE_MODEL}" \
  --data_path "${DATA_PATH}" \
  --output_dir "${OUTPUT_DIR:-outputs/llama/genft-alpaca}"

