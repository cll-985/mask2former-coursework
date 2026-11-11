#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "Usage: bash offline_tools/run_experiment.sh EXP_NAME CONFIG NUM_GPUS SEED [DETECTRON2_OPTS...]"
  exit 2
fi

EXP_NAME="$1"
CONFIG="$2"
NUM_GPUS="$3"
SEED="$4"
shift 4

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

if [[ -n "${TRAIN_ENTRY:-}" ]]; then
  TRAIN_SCRIPT="$TRAIN_ENTRY"
elif [[ -f "train_cardd.py" ]]; then
  TRAIN_SCRIPT="train_cardd.py"
else
  TRAIN_SCRIPT="train_net.py"
fi

if [[ ! -f "$CONFIG" ]]; then
  echo "Config not found: $CONFIG"
  exit 2
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${PROJECT_ROOT}/output_offline/${EXP_NAME}/${STAMP}_seed${SEED}"
mkdir -p "$RUN_DIR"

cp "$CONFIG" "$RUN_DIR/config_input.yaml"
python -V > "$RUN_DIR/python_version.txt" 2>&1
python -m pip freeze > "$RUN_DIR/pip_freeze.txt" 2>&1 || true
nvidia-smi > "$RUN_DIR/nvidia_smi.txt" 2>&1 || true
git rev-parse HEAD > "$RUN_DIR/git_commit.txt" 2>&1 || true
git status --short > "$RUN_DIR/git_status.txt" 2>&1 || true
sha256sum "$CONFIG" "$TRAIN_SCRIPT" > "$RUN_DIR/input_sha256.txt" 2>&1 || true

{
  echo "experiment=$EXP_NAME"
  echo "config=$CONFIG"
  echo "num_gpus=$NUM_GPUS"
  echo "seed=$SEED"
  echo "train_script=$TRAIN_SCRIPT"
  echo "run_dir=$RUN_DIR"
  printf 'extra_opts='
  printf '%q ' "$@"
  echo
} > "$RUN_DIR/run_meta.txt"

set +e
python "$TRAIN_SCRIPT" \
  --num-gpus "$NUM_GPUS" \
  --config-file "$CONFIG" \
  OUTPUT_DIR "$RUN_DIR" \
  SEED "$SEED" \
  "$@" 2>&1 | tee "$RUN_DIR/console.log"
EXIT_CODE=${PIPESTATUS[0]}
set -e

echo "$EXIT_CODE" > "$RUN_DIR/exit_code.txt"
python offline_tools/summarize_metrics.py \
  --metrics "$RUN_DIR/metrics.json" \
  --output "$RUN_DIR/metrics_summary.json" || true

echo "Run directory: $RUN_DIR"
exit "$EXIT_CODE"
