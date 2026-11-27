#!/usr/bin/env bash
# Every experiment that appears in reports/. Runs sequentially on one GPU.
#   bash scripts/run_all.sh            (≈ 1 h for ViT-B variants, ViT-L adds several hours)
set -euo pipefail
cd "$(dirname "$0")/.."
FOLDS=0,1,2,3,4,5,6,7,8,9

# Baselines that never see our data: one run evaluates all 10 folds.
uv run scripts/train.py variant=zero_shot
uv run scripts/train.py variant=external_only

# Trained with our data: one run per held-out fold.
uv run scripts/train.py -m variant=ours_only fold=$FOLDS
uv run scripts/train.py -m variant=ours_external fold=$FOLDS

# Strongest system: larger backbone.
if [[ "${WITH_LARGE:-1}" == "1" ]]; then
  uv run scripts/train.py -m variant=ours_external model=dinov3_vitl fold=$FOLDS
fi

uv run scripts/make_report.py
