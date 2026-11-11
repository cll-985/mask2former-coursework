#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: bash offline_tools/start_formal_baseline.sh SEED"
  echo "Recommended seeds: 42, 3407, 2026"
  exit 2
fi

SEED="$1"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

if [[ ! "$SEED" =~ ^[0-9]+$ ]]; then
  echo "Invalid seed: $SEED"
  exit 2
fi

# Do not start a second training process on the same single-GPU container.
if command -v pgrep >/dev/null 2>&1; then
  RUNNING="$(pgrep -af '[t]rain_cardd.py' || true)"
  if [[ -n "$RUNNING" ]]; then
    echo "Refusing to start seed $SEED: train_cardd.py is already running:"
    echo "$RUNNING"
    echo "Inspect these processes before stopping or starting anything."
    exit 3
  fi
fi

LAUNCH_ROOT="${PROJECT_ROOT}/output_offline/B00_formal_R50/launcher"
mkdir -p "$LAUNCH_ROOT"
STAMP="$(date +%Y%m%d_%H%M%S)"

if command -v tmux >/dev/null 2>&1; then
  SESSION="baseline${SEED}"
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "tmux session already exists: $SESSION"
    exit 4
  fi
  tmux new-session -d -s "$SESSION" \
    "cd '$PROJECT_ROOT' && bash offline_tools/run_formal_baseline.sh '$SEED'"
  echo "Started seed $SEED in tmux session: $SESSION"
  echo "Attach with: tmux attach -t $SESSION"
  exit 0
fi

if ! command -v nohup >/dev/null 2>&1; then
  echo "Neither tmux nor nohup is available."
  echo "Run in foreground: bash offline_tools/run_formal_baseline.sh $SEED"
  exit 5
fi

LAUNCH_LOG="${LAUNCH_ROOT}/${STAMP}_seed${SEED}.log"
PID_FILE="${LAUNCH_ROOT}/${STAMP}_seed${SEED}.pid"

nohup bash offline_tools/run_formal_baseline.sh "$SEED" \
  >"$LAUNCH_LOG" 2>&1 &
LAUNCH_PID=$!
echo "$LAUNCH_PID" >"$PID_FILE"

sleep 3
if ! kill -0 "$LAUNCH_PID" 2>/dev/null; then
  echo "Launcher exited early. Inspect: $LAUNCH_LOG"
  tail -50 "$LAUNCH_LOG" || true
  exit 6
fi

echo "tmux is unavailable; started seed $SEED with nohup."
echo "launcher_pid=$LAUNCH_PID"
echo "launcher_log=$LAUNCH_LOG"
echo "pid_file=$PID_FILE"
echo "Monitor launcher: tail -f '$LAUNCH_LOG'"

