#!/usr/bin/env bash
set -euo pipefail

for task in cola sst2 mrpc stsb qqp mnli qnli rte; do
  ADAPTER_SUBFOLDER="${task}/best_model" bash scripts/test_glue.sh "${task}"
done

