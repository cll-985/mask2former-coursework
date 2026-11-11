#!/usr/bin/env bash
set -u

PROJECT=/homex/cll2/cll3/Mask2Former-main
STAMP=$(date +%Y%m%d_%H%M%S)
RUN="$PROJECT/output_offline/Q3_empirical_evidence/M2_efficiency_${STAMP}"
mkdir -p "$RUN"
printf '%s\n' "$RUN" > "$PROJECT/output_offline/Q3_empirical_evidence/latest_m2_efficiency_run.txt"

LOG="$RUN/launcher.log"
exec >>"$LOG" 2>&1
echo "run=$RUN"
echo "Waiting for physical GPU 3: memory <= 3000 MiB and utilization <= 10% for 5 consecutive checks."

stable=0
while [ "$stable" -lt 5 ]; do
    line=$(nvidia-smi --query-gpu=memory.used,utilization.gpu,temperature.gpu \
        --format=csv,noheader,nounits --id=3 | head -1)
    used=$(printf '%s' "$line" | cut -d, -f1 | tr -d ' ')
    util=$(printf '%s' "$line" | cut -d, -f2 | tr -d ' ')
    temp=$(printf '%s' "$line" | cut -d, -f3 | tr -d ' ')
    if [ "$used" -le 3000 ] && [ "$util" -le 10 ]; then
        stable=$((stable + 1))
    else
        stable=0
    fi
    echo "$(date '+%F %T') used_mib=$used util_percent=$util temp_c=$temp stable=$stable/5"
    [ "$stable" -ge 5 ] || sleep 60
done

echo "GPU 3 is stably idle; starting the paper efficiency benchmark."
docker exec m2f_gpu3 bash -lc "
    cd /workspace/Mask2Former-main &&
    python offline_tools/benchmark_inference_efficiency.py \\
      --config-file configs/cardd/maskformer2_R50_cardd_baseline.yaml \\
      --weights-primary output_offline/B00_formal_R50/20260822_073123_seed3407/model_final.pth \\
      --weights-secondary output_offline/B00_formal_R50/20260819_133145_seed42/model_final.pth \\
      --dataset cardd_val \\
      --layer 3 \\
      --match-iou 0.5 \\
      --warmup 20 \\
      --max-images 200 \\
      --seed 20261009 \\
      --output /workspace/Mask2Former-main/output_offline/Q3_empirical_evidence/M2_efficiency_${STAMP}/efficiency_report.json \\
      > /workspace/Mask2Former-main/output_offline/Q3_empirical_evidence/M2_efficiency_${STAMP}/console.log 2>&1
"
status=$?
printf '%s\n' "$status" > "$RUN/exit_code.txt"
echo "benchmark_exit=$status"
exit "$status"
