#!/usr/bin/env bash
set -euo pipefail

BASE_MODEL="${BASE_MODEL:-baffo32/decapoda-research-llama-7B-hf}"
MODEL_SOURCE="${MODEL_SOURCE:-hf}"
ADAPTER_PATH="${ADAPTER_PATH:-modelscope://YOUR_NAMESPACE/GenFT-LLaMA-7B-Alpaca}"
ADAPTER_SUBFOLDER="${ADAPTER_SUBFOLDER:-genft}"
INSTRUCTION="${1:-Tell me about alpacas.}"

python examples/test_llama_generate.py \
  --model_source "${MODEL_SOURCE}" \
  --model_name_or_path "${BASE_MODEL}" \
  --adapter_path "${ADAPTER_PATH}" \
  --adapter_subfolder "${ADAPTER_SUBFOLDER}" \
  --instruction "${INSTRUCTION}"
