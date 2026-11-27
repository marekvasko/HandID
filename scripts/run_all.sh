#!/usr/bin/env bash
# Every experiment that appears in reports/. Runs sequentially on one GPU and skips
# runs that already have results, so it can be restarted after an interruption.
#   bash scripts/run_all.sh                 # ViT-B experiments (~4 h on an RTX 5090)
#   WITH_LARGE=1 bash scripts/run_all.sh    # + ViT-L (several more hours)
set -euo pipefail
cd "$(dirname "$0")/.."

run() {  # run <variant> <model> <fold> [extra hydra overrides...]
  local variant=$1 model=$2 fold=$3; shift 3
  local name=${RUN_SUFFIX:-}
  if [[ -f "outputs/${variant}/${model}${name}/fold${fold}/results.json" ]]; then
    echo "skip ${variant}/${model}${name}/fold${fold} (done)"; return
  fi
  uv run scripts/train.py variant="$variant" model="$model" fold="$fold" "$@"
}

# Baselines that never see our data: a single run evaluates all 10 folds.
run zero_shot     dinov3_vitb 0
run external_only dinov3_vitb 0

# Trained with our data: one run per held-out fold.
for fold in {0..9}; do run ours_only     dinov3_vitb "$fold"; done
for fold in {0..9}; do run ours_external dinov3_vitb "$fold"; done

# Ablation: background removed with a landmark-based hand mask.
if [[ "${WITH_MASK:-1}" == "1" ]]; then
  for fold in {0..9}; do
    RUN_SUFFIX=_masked run ours_external dinov3_vitb "$fold" input=masked
  done
fi

# Strongest system: larger backbone.
if [[ "${WITH_LARGE:-0}" == "1" ]]; then
  for fold in {0..9}; do run ours_external dinov3_vitl "$fold"; done
fi

uv run scripts/make_report.py
