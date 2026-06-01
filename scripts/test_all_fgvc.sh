#!/usr/bin/env bash
set -euo pipefail

for dataset in CUB_200_2011 nabirds OxfordFlower StanfordCars StanfordDogs; do
  bash scripts/test_fgvc.sh "${dataset}"
done

