#!/usr/bin/env bash
set -euo pipefail

for dataset in cifar caltech101 dtd oxford_flowers102 oxford_iiit_pet svhn sun397 patch_camelyon eurosat resisc45 diabetic_retinopathy clevr_count clevr_dist dmlab kitti dsprites_loc dsprites_ori smallnorb_azi smallnorb_ele; do
  bash scripts/test_vtab.sh "${dataset}"
done

