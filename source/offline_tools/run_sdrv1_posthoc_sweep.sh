#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

GPU_INDEX="${GPU_INDEX:-0}"
MAX_USED_MB="${MAX_USED_MB:-3000}"
MAX_UTIL_PERCENT="${MAX_UTIL_PERCENT:-20}"
POLL_SECONDS="${POLL_SECONDS:-60}"
REQUIRED_STABLE_CHECKS="${REQUIRED_STABLE_CHECKS:-5}"

CONFIG="configs/cardd/maskformer2_R50_cardd_sdr_v1_headisolated.yaml"
WEIGHTS="output_offline/P3_SDRv1_headisolated_formal_R50/20260831_062520_seed42/model_final.pth"
SWEEP_ROOT="output_offline/P3_SDRv1_posthoc_sweep_seed42"
RESULTS_TSV="$SWEEP_ROOT/results.tsv"
mkdir -p "$SWEEP_ROOT"

if [[ ! -f "$CONFIG" || ! -f "$WEIGHTS" ]]; then
  echo "Missing config or weights" >&2
  exit 2
fi

echo "Waiting for GPU ${GPU_INDEX} to be idle before the sweep"
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

  printf '%s used_mb=%s util_percent=%s train_busy=%s stable=%s/%s\n' \
    "$(date '+%F %T')" "$used_mb" "$util_percent" "$train_busy" \
    "$stable_checks" "$REQUIRED_STABLE_CHECKS"

  if (( stable_checks < REQUIRED_STABLE_CHECKS )); then
    sleep "$POLL_SECONDS"
  fi
done

CASES=(
  "off False 0 0.05"
  "h2_m005 True 2 0.05"
  "h4_m005 True 4 0.05"
  "h8_m005 True 8 0.05"
  "h2_m010 True 2 0.10"
  "h2_m015 True 2 0.15"
  "h4_m010 True 4 0.10"
)

printf 'name\tenabled\trescue_heads\tmargin\tAP\tAP50\tAP75\tAPs\tAPm\tAPl\texit_code\n' > "$RESULTS_TSV"

for spec in "${CASES[@]}"; do
  read -r name enabled rescue_heads margin <<< "$spec"
  output_dir="$SWEEP_ROOT/$name"
  mkdir -p "$output_dir"

  echo "START name=$name enabled=$enabled rescue_heads=$rescue_heads margin=$margin"

  if [[ -f "$output_dir/exit_code.txt" ]] && [[ "$(cat "$output_dir/exit_code.txt")" == "0" ]]; then
    echo "SKIP completed case $name"
  else
    set +e
    CUDA_VISIBLE_DEVICES="$GPU_INDEX" python train_cardd.py \
      --num-gpus 1 \
      --config-file "$CONFIG" \
      --eval-only \
      MODEL.WEIGHTS "$WEIGHTS" \
      OUTPUT_DIR "$output_dir" \
      SEED 42 \
      MODEL.MASK_FORMER.SDR.ENABLED "$enabled" \
      MODEL.MASK_FORMER.SDR.RESCUE_HEADS "$rescue_heads" \
      MODEL.MASK_FORMER.SDR.UNCERTAINTY_MARGIN "$margin" \
      2>&1 | tee "$output_dir/console.log"
    exit_code=${PIPESTATUS[0]}
    set -e
    printf '%s\n' "$exit_code" > "$output_dir/exit_code.txt"
    if (( exit_code != 0 )); then
      echo "FAILED name=$name exit_code=$exit_code" >&2
      exit "$exit_code"
    fi
  fi

  metrics="$(grep -A2 'copypaste: Task: segm' "$output_dir/console.log" | tail -1 | sed -E 's/^.*copypaste: //')"
  printf '%s\t%s\t%s\t%s\t%s\t0\n' \
    "$name" "$enabled" "$rescue_heads" "$margin" "${metrics//,/\t}" \
    >> "$RESULTS_TSV"
  echo "END name=$name segm=$metrics"
done

echo "SWEEP_COMPLETE"
column -t -s $'\t' "$RESULTS_TSV" 2>/dev/null || cat "$RESULTS_TSV"
