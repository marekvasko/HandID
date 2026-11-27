#!/usr/bin/env bash
# GPU job queue. Runs the experiments listed in configs/queue.txt one after another.
# The list is re-read before every run, so lines can be added/reordered while it runs.
# A run is skipped when its results.json exists, so the queue can be restarted at any time.
#   setsid nohup bash scripts/queue.sh > outputs_dev/queue.log 2>&1 &
#
# queue.txt line format:  <variant> <model> <fold> <input:plain|masked> [hydra overrides...]
set -uo pipefail
cd "$(dirname "$0")/.."
declare -A failed=()

next_job() {
  while read -r variant model fold input extra; do
    [[ -z "${variant:-}" || "$variant" == \#* ]] && continue
    local suffix=""; [[ $input == masked ]] && suffix=_masked
    local dir="outputs/${variant}/${model}${suffix}/fold${fold}"
    [[ -f "$dir/results.json" || -n "${failed[$dir]:-}" ]] && continue
    echo "$dir|$variant $model $fold $input ${extra:-}"; return
  done < configs/queue.txt
}

while job=$(next_job) && [[ -n "$job" ]]; do
  dir=${job%%|*}; read -r variant model fold input extra <<< "${job#*|}"
  echo "=== $(date '+%m-%d %H:%M') $dir"
  # shellcheck disable=SC2086
  uv run scripts/train.py variant="$variant" model="$model" fold="$fold" input="$input" ${extra:-} \
    || { echo "!!! FAILED $dir"; failed[$dir]=1; }
done
echo "queue empty $(date '+%m-%d %H:%M')"
