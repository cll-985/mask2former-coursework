#!/usr/bin/env bash
set -u

PROJECT_ROOT=${PROJECT_ROOT:-/workspace/Mask2Former-main}
OUTPUT_REL=${OUTPUT_REL:-output_offline/P11_trajectory_router/train_seed42_layer3_final9}
WEIGHTS_REL=${WEIGHTS_REL:-output_offline/B00_formal_R50/20260819_133145_seed42/model_final.pth}
CONFIG_REL=${CONFIG_REL:-configs/cardd/maskformer2_R50_cardd_baseline.yaml}
GPU_INDEX=${GPU_INDEX:-0}
MAX_USED_MB=${MAX_USED_MB:-1500}
MAX_UTIL_PERCENT=${MAX_UTIL_PERCENT:-10}
REQUIRED_STABLE_CHECKS=${REQUIRED_STABLE_CHECKS:-5}
POLL_SECONDS=${POLL_SECONDS:-60}

cd "$PROJECT_ROOT" || exit 2
mkdir -p "$OUTPUT_REL"

echo "Waiting for GPU ${GPU_INDEX} to be idle before train-set trajectory export"
stable=0
while :; do
    values=$(nvidia-smi --query-gpu=memory.used,utilization.gpu \
        --format=csv,noheader,nounits -i "$GPU_INDEX" 2>/dev/null | head -1)
    used_mb=$(printf '%s' "$values" | cut -d, -f1 | tr -d ' ')
    util_percent=$(printf '%s' "$values" | cut -d, -f2 | tr -d ' ')
    train_busy=0
    if pgrep -af '[t]rain_cardd.py|[e]valuate_layer_consensus.py' >/dev/null; then
        train_busy=1
    fi
    if [ -n "$used_mb" ] && [ -n "$util_percent" ] && \
       [ "$used_mb" -le "$MAX_USED_MB" ] && \
       [ "$util_percent" -le "$MAX_UTIL_PERCENT" ] && \
       [ "$train_busy" -eq 0 ]; then
        stable=$((stable + 1))
    else
        stable=0
    fi
    printf '%s used_mb=%s util_percent=%s train_busy=%s stable=%s/%s\n' \
        "$(date '+%F %T')" "${used_mb:-NA}" "${util_percent:-NA}" \
        "$train_busy" "$stable" "$REQUIRED_STABLE_CHECKS"
    [ "$stable" -ge "$REQUIRED_STABLE_CHECKS" ] && break
    sleep "$POLL_SECONDS"
done

echo "Starting cardd_train layer3/layer9 export"
CUDA_VISIBLE_DEVICES="$GPU_INDEX" python offline_tools/evaluate_layer_consensus.py \
    --config-file "$CONFIG_REL" \
    --weights "$WEIGHTS_REL" \
    --dataset cardd_train \
    --output-dir "$OUTPUT_REL" \
    --layers 3,9 \
    --seed 42 \
    > "$OUTPUT_REL/console.log" 2>&1
status=$?
printf '%s\n' "$status" > "$OUTPUT_REL/exit_code.txt"
echo "Export finished with exit=${status}"
exit "$status"
