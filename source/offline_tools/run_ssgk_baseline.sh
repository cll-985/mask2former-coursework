#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: bash offline_tools/run_ssgk_baseline.sh SEED"
  echo "Predeclared seeds: 42, 3407, 2026"
  exit 2
fi

SEED="$1"
case "$SEED" in
  42|3407|2026) ;;
  *)
    echo "Refusing non-predeclared seed: $SEED"
    exit 2
    ;;
esac

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

CONFIG="configs/ssgk/maskformer2_R50_ssgk_baseline.yaml"
WEIGHT="datasets/pretrained/R-50.pkl"
SSGK_ROOT="${SSGK_ROOT:-${PROJECT_ROOT}/datasets/coco_ssgk}"
TRAIN_JSON="${SSGK_ROOT}/annotations_9cls/instances_train2017.json"
VAL_JSON="${SSGK_ROOT}/annotations_9cls/instances_val2017.json"

python offline_tools/preflight.py \
  --config "$CONFIG" \
  --required-file train_ssgk.py \
  --required-file "$WEIGHT" \
  --required-file "$TRAIN_JSON" \
  --required-file "$VAL_JSON"

SSGK_ROOT="$SSGK_ROOT" TRAIN_ENTRY=train_ssgk.py \
bash offline_tools/run_experiment.sh \
  "E01_SSGK_external_R50" \
  "$CONFIG" \
  1 \
  "$SEED"

