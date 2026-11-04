#!/usr/bin/env bash
# Run this on the GPU pod, inside tmux (`tmux new -s tpbench`) or with nohup,
# so an SSH disconnect doesn't kill a multi-hour sweep:
#
#   tmux new -s tpbench
#   ./run_session.sh
#   # detach: Ctrl-b d ; reattach later: tmux attach -t tpbench
set -euo pipefail

HOME_DIR="${TPB_HOME:-/workspace}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source "$HOME_DIR/venv/bin/activate"
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

SESSION_DIR="$HOME_DIR/results/session_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$SESSION_DIR"

nvidia-smi -q > "$SESSION_DIR/nvidia-smi.txt"
python "$SCRIPT_DIR/idle_power.py" --seconds 60 --out "$SESSION_DIR/idle_power.json"

python "$SCRIPT_DIR/run_sweep.py" --config "$SCRIPT_DIR/sweep_config.yaml" --home "$HOME_DIR" \
    2>&1 | tee "$SESSION_DIR/sweep.log"

echo
echo "Sweep finished. Copy results off before stopping/terminating this pod -"
echo "container disk is not persisted. For example, from your own machine:"
echo "  runpodctl send $HOME_DIR/results"
echo
echo "To stop the pod automatically once copied, uncomment the line below"
echo "and re-run, or run it manually once you have the results:"
# runpodctl stop pod "$RUNPOD_POD_ID"
