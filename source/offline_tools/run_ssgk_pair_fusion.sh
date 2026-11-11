#!/usr/bin/env bash
set -u

# Frozen SSGK output-space fusion protocol.
# Runs the three pre-specified seed pairs sequentially using the saved final
# COCO predictions.  No training or validation-time parameter search occurs.

PROJECT_ROOT=${PROJECT_ROOT:-/workspace/Mask2Former-main}
BASELINE_ROOT=${BASELINE_ROOT:-output_offline/E01_SSGK_external_R50}
OUTPUT_ROOT=${OUTPUT_ROOT:-output_offline/E02_SSGK_pair_fusion_R50}
GT_JSON=${GT_JSON:-datasets/coco_ssgk/annotations_9cls/instances_val2017.json}
MATCH_IOU=${MATCH_IOU:-0.5}
MAX_DETS=${MAX_DETS:-100}
RULES=${RULES:-mask_nms,consensus}

cd "$PROJECT_ROOT" || exit 90
mkdir -p "$OUTPUT_ROOT"

SEED42="$BASELINE_ROOT/20261009_043435_seed42/inference/coco_instances_results.json"
SEED3407="$BASELINE_ROOT/20261010_230600_seed3407/inference/coco_instances_results.json"
SEED2026="$BASELINE_ROOT/20261012_161755_seed2026/inference/coco_instances_results.json"

for required in "$GT_JSON" "$SEED42" "$SEED3407" "$SEED2026"; do
  if [ ! -s "$required" ]; then
    echo "missing_or_empty=$required" >&2
    exit 91
  fi
done

run_pair() {
  name=$1
  base=$2
  other=$3
  pair_dir="$OUTPUT_ROOT/$name"
  mkdir -p "$pair_dir"

  if [ -f "$pair_dir/exit_code.txt" ] && [ "$(cat "$pair_dir/exit_code.txt")" = "0" ]; then
    echo "SKIP name=$name reason=already_complete"
    return 0
  fi

  echo "START name=$name time=$(date -Is)"
  python offline_tools/compare_fusion_rules.py \
    --base-results "$base" \
    --other-results "$other" \
    --gt-json "$GT_JSON" \
    --output-dir "$pair_dir" \
    --match-iou "$MATCH_IOU" \
    --max-dets "$MAX_DETS" \
    --rules "$RULES" \
    > "$pair_dir/console.log" 2>&1
  code=$?
  printf '%s\n' "$code" > "$pair_dir/exit_code.txt"
  echo "END name=$name exit=$code time=$(date -Is)"
  return "$code"
}

run_pair seed42_seed3407 "$SEED42" "$SEED3407" || exit $?
run_pair seed42_seed2026 "$SEED42" "$SEED2026" || exit $?
run_pair seed3407_seed2026 "$SEED3407" "$SEED2026" || exit $?

printf 'match_iou\t%s\nmax_dets\t%s\nrules\t%s\n' \
  "$MATCH_IOU" "$MAX_DETS" "$RULES" > "$OUTPUT_ROOT/frozen_protocol.tsv"
printf '0\n' > "$OUTPUT_ROOT/exit_code.txt"
echo "ALL_PAIRS_COMPLETE time=$(date -Is)"
