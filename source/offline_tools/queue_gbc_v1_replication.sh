#!/usr/bin/env bash
# Wait for the dedicated GPU3 container's single visible device to be idle,
# then run one unchanged GBC-v1 replication. Never interrupts another process.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

SEED="${1:-3407}"
EXPECTED_GPU_BUS_ID="${EXPECTED_GPU_BUS_ID:-00000000:AF:00.0}"
if [[ "$SEED" != "3407" ]]; then
  echo "This guarded queue is intentionally limited to seed3407." >&2
  exit 2
fi

EXP_NAME="P4_GBCv1_formal_R50"
CONFIG="configs/cardd/maskformer2_R50_cardd_gbc_v1.yaml"
FORMAL_ROOT="output_offline/$EXP_NAME"
QUEUE_ROOT="output_offline/P4_GBCv1_seed3407_queue"
mkdir -p "$FORMAL_ROOT" "$QUEUE_ROOT"

exec 9>"$QUEUE_ROOT/queue.lock"
if ! flock -n 9; then
  echo "Another GBC-v1 replication queue is already active." >&2
  exit 3
fi

existing_run() {
  find "$FORMAL_ROOT" -mindepth 1 -maxdepth 1 -type d \
    -name "*_seed${SEED}" -print -quit
}

if [[ -n "$(existing_run)" ]]; then
  echo "A seed${SEED} run directory already exists; refusing to duplicate it." >&2
  exit 4
fi

bus_id="$(nvidia-smi -i 0 --query-gpu=pci.bus_id --format=csv,noheader | head -1 | tr -d '[:space:]')"
if [[ "$bus_id" != "$EXPECTED_GPU_BUS_ID" ]]; then
  echo "Expected GPU bus $EXPECTED_GPU_BUS_ID, got $bus_id; refusing to launch." >&2
  exit 5
fi

echo "queue_pid=$$ seed=$SEED gpu_bus=$bus_id"
echo "idle_gate=memory<=1000MiB, utilization<=10%, five consecutive 60s checks"
echo "max_wait=72h"
sha256sum "$CONFIG" mask2former/config.py mask2former/maskformer_model.py \
  mask2former/modeling/criterion.py

deadline=$(( $(date +%s) + 72 * 3600 ))
stable=0
while (( stable < 5 )); do
  if (( $(date +%s) >= deadline )); then
    echo "72-hour wait expired without launching." >&2
    exit 6
  fi
  if [[ -n "$(existing_run)" ]]; then
    echo "A seed${SEED} run appeared during the wait; refusing to duplicate it." >&2
    exit 7
  fi

  read -r used_mb util_percent < <(
    nvidia-smi -i 0 --query-gpu=memory.used,utilization.gpu \
      --format=csv,noheader,nounits | head -1 | tr -d ' ' | tr ',' ' '
  )
  if pgrep -f '[t]rain_cardd.py' >/dev/null; then
    train_busy=1
  else
    train_busy=0
  fi
  if [[ "$used_mb" =~ ^[0-9]+$ && "$util_percent" =~ ^[0-9]+$ ]] &&
     (( used_mb <= 1000 && util_percent <= 10 && train_busy == 0 )); then
    stable=$((stable + 1))
  else
    stable=0
  fi
  printf '%s used_mb=%s util=%s train_busy=%s stable=%s/5\n' \
    "$(date '+%F %T')" "$used_mb" "$util_percent" "$train_busy" "$stable"
  if (( stable < 5 )); then
    sleep 60
  fi
done

echo "$(date '+%F %T') GPU idle gate passed; launching $EXP_NAME seed$SEED"
CUDA_VISIBLE_DEVICES=0 TRAIN_ENTRY=train_cardd.py \
  bash offline_tools/run_experiment.sh "$EXP_NAME" "$CONFIG" 1 "$SEED"
