#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

SEED="${1:-42}"
GPU_INDEX="${GPU_INDEX:-0}"
MAX_USED_MB="${MAX_USED_MB:-3000}"
MAX_UTIL_PERCENT="${MAX_UTIL_PERCENT:-20}"
POLL_SECONDS="${POLL_SECONDS:-60}"
REQUIRED_STABLE_CHECKS="${REQUIRED_STABLE_CHECKS:-5}"

EXP_NAME="P3_SDRv1_headisolated_formal_R50"
CONFIG="configs/cardd/maskformer2_R50_cardd_sdr_v1_headisolated.yaml"
QUEUE_ROOT="output_offline/${EXP_NAME}_queue"
FORMAL_ROOT="output_offline/${EXP_NAME}"
mkdir -p "$QUEUE_ROOT" "$FORMAL_ROOT"

echo "queue_pid=$$"
echo "seed=$SEED"
echo "gpu_index=$GPU_INDEX"
echo "idle_gate=memory<=${MAX_USED_MB}MiB, utilization<=${MAX_UTIL_PERCENT}%, ${REQUIRED_STABLE_CHECKS} consecutive checks"

stable_checks=0
while (( stable_checks < REQUIRED_STABLE_CHECKS )); do
  used_mb="$(nvidia-smi -i "$GPU_INDEX" --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d '[:space:]')"
  util_percent="$(nvidia-smi -i "$GPU_INDEX" --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1 | tr -d '[:space:]')"

  if pgrep -f '[t]rain_cardd.py' >/dev/null; then
    train_busy=1
  else
    train_busy=0
  fi

  if (( train_busy == 0 && used_mb <= MAX_USED_MB && util_percent <= MAX_UTIL_PERCENT )); then
    stable_checks=$((stable_checks + 1))
  else
    stable_checks=0
  fi

  printf '%s used_mb=%s util_percent=%s train_cardd_busy=%s stable=%s/%s\n' \
    "$(date '+%F %T')" "$used_mb" "$util_percent" "$train_busy" \
    "$stable_checks" "$REQUIRED_STABLE_CHECKS"

  if (( stable_checks < REQUIRED_STABLE_CHECKS )); then
    sleep "$POLL_SECONDS"
  fi
done

echo "$(date '+%F %T') idle gate passed; starting ${EXP_NAME} seed ${SEED}"

set +e
CUDA_VISIBLE_DEVICES="$GPU_INDEX" TRAIN_ENTRY=train_cardd.py \
  bash offline_tools/run_experiment.sh \
    "$EXP_NAME" \
    "$CONFIG" \
    1 \
    "$SEED"
exit_code=$?
set -e

latest_run="$(find "$FORMAL_ROOT" -mindepth 1 -maxdepth 1 -type d -name "*_seed${SEED}" | sort | tail -1)"
if [[ -n "$latest_run" ]]; then
  printf '%s\n' "$latest_run" > "$FORMAL_ROOT/latest_seed${SEED}_run.txt"
  echo "latest_run=$latest_run"
fi

echo "formal_exit_code=$exit_code"
exit "$exit_code"
