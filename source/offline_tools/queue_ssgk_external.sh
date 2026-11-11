#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

SEEDS="${SSGK_QUEUE_SEEDS:-42 3407 2026}"
MAX_USED_MB="${SSGK_MAX_USED_MB:-1500}"
MAX_UTIL_PERCENT="${SSGK_MAX_UTIL_PERCENT:-10}"
STABLE_REQUIRED="${SSGK_STABLE_REQUIRED:-5}"
POLL_SECONDS="${SSGK_POLL_SECONDS:-60}"
EXPERIMENT_ROOT="${PROJECT_ROOT}/output_offline/E01_SSGK_external_R50"

mkdir -p "$EXPERIMENT_ROOT"

log() {
  echo "$(date '+%Y-%m-%d %H:%M:%S') $*"
}

completed_seed() {
  local seed="$1"
  local code_file
  while IFS= read -r code_file; do
    if [[ "$(cat "$code_file" 2>/dev/null || true)" == "0" ]]; then
      return 0
    fi
  done < <(find "$EXPERIMENT_ROOT" -mindepth 2 -maxdepth 2 \
    -path "*_seed${seed}/exit_code.txt" -type f -print 2>/dev/null | sort)
  return 1
}

wait_for_idle_gpu() {
  local stable=0
  while (( stable < STABLE_REQUIRED )); do
    local line used util
    line="$(nvidia-smi --query-gpu=memory.used,utilization.gpu \
      --format=csv,noheader,nounits 2>/dev/null | head -1 || true)"
    used="$(echo "$line" | cut -d, -f1 | tr -d ' ')"
    util="$(echo "$line" | cut -d, -f2 | tr -d ' ')"
    if [[ "$used" =~ ^[0-9]+$ && "$util" =~ ^[0-9]+$ ]] && \
       (( used <= MAX_USED_MB && util <= MAX_UTIL_PERCENT )) && \
       ! pgrep -af '[t]rain_ssgk.py' >/dev/null; then
      stable=$((stable + 1))
    else
      stable=0
    fi
    log "waiting used_mb=${used:-NA} util_percent=${util:-NA} stable=${stable}/${STABLE_REQUIRED}"
    if (( stable < STABLE_REQUIRED )); then
      sleep "$POLL_SECONDS"
    fi
  done
}

log "queue_start seeds=${SEEDS} max_used_mb=${MAX_USED_MB} max_util=${MAX_UTIL_PERCENT}"
for seed in $SEEDS; do
  if completed_seed "$seed"; then
    log "skip_completed seed=${seed}"
    continue
  fi
  wait_for_idle_gpu
  log "run_start seed=${seed}"
  set +e
  bash offline_tools/run_ssgk_baseline.sh "$seed"
  code=$?
  set -e
  log "run_end seed=${seed} exit=${code}"
  if (( code != 0 )); then
    log "queue_stop_failed seed=${seed}"
    exit "$code"
  fi
done
log "queue_complete"

