#!/usr/bin/env bash
set -euo pipefail

DATASET="${1:-CUB_200_2011}"
DATASET_DIR="${DATASET_DIR:-data/fgvc/}"
VIT_CKPT="${VIT_CKPT:-pretrained/ViT-B_16.npz}"

if [ -z "${CLASS_NUM:-}" ]; then
  case "${DATASET}" in
    CUB_200_2011) CLASS_NUM=200 ;;
    nabirds) CLASS_NUM=555 ;;
    OxfordFlower) CLASS_NUM=102 ;;
    StanfordCars) CLASS_NUM=196 ;;
    StanfordDogs) CLASS_NUM=120 ;;
    *)
      echo "Unknown FGVC dataset '${DATASET}'. Set CLASS_NUM manually." >&2
      exit 1
      ;;
  esac
fi

python examples/train_image.py \
  --benchmark fgvc \
  --dataset "${DATASET}" \
  --class_num "${CLASS_NUM}" \
  --dataset_dir "${DATASET_DIR}" \
  --model_checkpoint "${VIT_CKPT}"
