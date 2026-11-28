#!/usr/bin/env bash
# Every experiment that appears in reports/, in priority order. Runs sequentially on one GPU,
# skips runs that already have results (restartable) and continues after a failed run.
#   bash scripts/run_all.sh
# Tip: run a *copy* of this file if you want to edit it while it runs (bash reads scripts lazily).
set -uo pipefail
cd "$(dirname "$0")/.."
ALL="0 1 2 3 4 5 6 7 8 9"   # full 10-fold cross-validation over people
ABL="0 1 2 3 4"             # ablations: paired comparison on the first 5 folds

run() {  # run <variant> <model> <fold> [input] [extra hydra overrides...]
  local variant=$1 model=$2 fold=$3 input=${4:-plain}; shift 4 2>/dev/null || shift $#
  local suffix=""; [[ $input == masked ]] && suffix=_masked
  local dir="outputs/${variant}/${model}${suffix}/fold${fold}"
  [[ -f "$dir/results.json" ]] && { echo "skip $dir"; return; }
  echo "=== $(date +%H:%M) $dir"
  uv run scripts/train.py variant="$variant" model="$model" fold="$fold" input="$input" "$@" \
    || echo "!!! FAILED $dir"
}

# 1. Baselines without our data (one run evaluates all 10 folds)            -> issue #2
for m in dinov3_vitb clip_vitb siglip2_vitb; do run zero_shot "$m" 0; done
run external_only dinov3_vitb 0

# 2. Main systems, 10 folds                                                  -> reports/task1, task2
for f in $ALL; do run ours_only     dinov3_vitb "$f"; done
for f in $ALL; do run ours_external dinov3_vitb "$f"; done

# 3. Session-only supervision, 10 folds                                      -> issue #3
for f in $ALL; do run ours_sessions          dinov3_vitb "$f"; done
for f in $ALL; do run ours_sessions_external dinov3_vitb "$f"; done

# 4. Ablations on 5 folds (compare with ours_external dinov3_vitb on the same folds)
for f in $ABL; do run ours_external dinov3_vitb "$f" masked; done              # issue #4
for m in siglip2_vitb clip_vitb dinov3_convnext_b; do                          # issue #2
  for f in $ABL; do run ours_external "$m" "$f"; done
done
for f in $ABL; do run ours_external_meta dinov3_vitb "$f"; done                # issue #5

# 5. Strongest system: ViT-L, longer schedule                                -> issue #6
for f in $ABL; do run ours_external dinov3_vitl "$f" plain train.epochs=20; done

uv run scripts/make_report.py
