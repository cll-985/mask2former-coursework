#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: bash offline_tools/run_formal_baseline.sh SEED"
  echo "Recommended seeds: 42, 3407, 2026"
  exit 2
fi

SEED="$1"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

CONFIG="configs/cardd/maskformer2_R50_cardd_baseline.yaml"
WEIGHT="datasets/pretrained/R-50.pkl"
TRAIN_JSON="CarDD_COCO/annotations/instances_train2017.json"

python offline_tools/preflight.py \
  --config "$CONFIG" \
  --required-file train_cardd.py \
  --required-file "$WEIGHT" \
  --required-file "$TRAIN_JSON"

TRAIN_ENTRY=train_cardd.py \
bash offline_tools/run_experiment.sh \
  "B00_formal_R50" \
  "$CONFIG" \
  1 \
  "$SEED"
