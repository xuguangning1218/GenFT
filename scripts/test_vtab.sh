#!/usr/bin/env bash
set -euo pipefail

DATASET="${1:-cifar}"
DATASET_DIR="${DATASET_DIR:-data/vtab-1k}"
VIT_CKPT="${VIT_CKPT:-pretrained/ViT-B_16.npz}"
ADAPTER_PATH="${ADAPTER_PATH:-outputs/}"
ADAPTER_SUBFOLDER="${ADAPTER_SUBFOLDER:-${DATASET}}"

if [ -z "${CLASS_NUM:-}" ]; then
  case "${DATASET}" in
    caltech101) CLASS_NUM=102 ;;
    cifar) CLASS_NUM=100 ;;
    clevr_count) CLASS_NUM=8 ;;
    clevr_dist) CLASS_NUM=6 ;;
    diabetic_retinopathy) CLASS_NUM=5 ;;
    dmlab) CLASS_NUM=6 ;;
    dsprites_loc) CLASS_NUM=16 ;;
    dsprites_ori) CLASS_NUM=16 ;;
    dtd) CLASS_NUM=47 ;;
    eurosat) CLASS_NUM=10 ;;
    kitti) CLASS_NUM=4 ;;
    oxford_flowers102) CLASS_NUM=102 ;;
    oxford_iiit_pet) CLASS_NUM=37 ;;
    patch_camelyon) CLASS_NUM=2 ;;
    resisc45) CLASS_NUM=45 ;;
    smallnorb_azi) CLASS_NUM=18 ;;
    smallnorb_ele) CLASS_NUM=9 ;;
    sun397) CLASS_NUM=397 ;;
    svhn) CLASS_NUM=10 ;;
    *)
      echo "Unknown VTAB dataset '${DATASET}'. Set CLASS_NUM manually." >&2
      exit 1
      ;;
  esac
fi

python examples/test_image.py \
  --benchmark vtab \
  --dataset "${DATASET}" \
  --class_num "${CLASS_NUM}" \
  --dataset_dir "${DATASET_DIR}" \
  --model_checkpoint "${VIT_CKPT}" \
  --adapter_path "${ADAPTER_PATH}" \
  --adapter_subfolder "${ADAPTER_SUBFOLDER}"
