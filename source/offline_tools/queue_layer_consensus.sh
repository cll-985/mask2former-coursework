#!/usr/bin/env bash
set -u

# Host-side safety queue for one full layer-consensus evaluation.  The script
# waits for a physical GPU to be genuinely idle before entering the container.

PROJECT_ROOT=${PROJECT_ROOT:-/homex/cll2/cll3/Mask2Former-main}
CONTAINER_ID=${CONTAINER_ID:-d63e2be9b37e}
PHYSICAL_GPU=${PHYSICAL_GPU:-1}
MAX_USED_MB=${MAX_USED_MB:-2000}
MAX_UTIL_PERCENT=${MAX_UTIL_PERCENT:-10}
STABLE_REQUIRED=${STABLE_REQUIRED:-3}
POLL_SECONDS=${POLL_SECONDS:-60}

RUN_NAME=${RUN_NAME:-seed42_layer3_final9}
OUTPUT_REL=${OUTPUT_REL:-output_offline/P9_layer_consensus_R50/${RUN_NAME}}
CONFIG_REL=${CONFIG_REL:-configs/cardd/maskformer2_R50_cardd_baseline.yaml}
WEIGHTS_REL=${WEIGHTS_REL:-output_offline/B00_formal_R50/20260819_133145_seed42/model_final.pth}
SEED=${SEED:-42}
LAYERS=${LAYERS:-3,9}

cd "$PROJECT_ROOT" || exit 90
mkdir -p "$OUTPUT_REL"
QUEUE_LOG="$OUTPUT_REL/queue.log"

if [ -f "$OUTPUT_REL/exit_code.txt" ]; then
  echo "$(date -Is) already_finished exit=$(cat "$OUTPUT_REL/exit_code.txt")" >> "$QUEUE_LOG"
  exit 0
fi

if docker exec "$CONTAINER_ID" bash -lc "pgrep -af '[e]valuate_layer_consensus.py'" >/dev/null 2>&1; then
  echo "$(date -Is) existing_layer_consensus_process; refusing duplicate" >> "$QUEUE_LOG"
  exit 91
fi

stable=0
echo "$(date -Is) waiting gpu=$PHYSICAL_GPU max_used_mb=$MAX_USED_MB max_util=$MAX_UTIL_PERCENT stable_required=$STABLE_REQUIRED" >> "$QUEUE_LOG"
while :; do
  line=$(nvidia-smi -i "$PHYSICAL_GPU" \
    --query-gpu=memory.used,utilization.gpu \
    --format=csv,noheader,nounits 2>>"$QUEUE_LOG") || {
      echo "$(date -Is) nvidia_smi_failed" >> "$QUEUE_LOG"
      sleep "$POLL_SECONDS"
      continue
    }
  IFS=',' read -r used util <<< "$line"
  used=$(echo "$used" | xargs)
  util=$(echo "$util" | xargs)
  if [ "$used" -le "$MAX_USED_MB" ] && [ "$util" -le "$MAX_UTIL_PERCENT" ]; then
    stable=$((stable + 1))
  else
    stable=0
  fi
  echo "$(date -Is) used_mb=$used util_percent=$util stable=$stable/$STABLE_REQUIRED" >> "$QUEUE_LOG"
  if [ "$stable" -ge "$STABLE_REQUIRED" ]; then
    break
  fi
  sleep "$POLL_SECONDS"
done

echo "$(date -Is) launching $RUN_NAME" >> "$QUEUE_LOG"
docker exec "$CONTAINER_ID" bash -lc "
cd /workspace/Mask2Former-main || exit 90
python offline_tools/evaluate_layer_consensus.py \\
  --config-file '$CONFIG_REL' \\
  --weights '$WEIGHTS_REL' \\
  --dataset cardd_val \\
  --output-dir '$OUTPUT_REL' \\
  --layers '$LAYERS' \\
  --seed '$SEED' \\
  > '$OUTPUT_REL/console.log' 2>&1
code=\$?
echo \"\$code\" > '$OUTPUT_REL/exit_code.txt'
exit \"\$code\"
"
code=$?
echo "$(date -Is) finished $RUN_NAME exit=$code" >> "$QUEUE_LOG"
exit "$code"
