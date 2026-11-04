#!/usr/bin/env bash
# Run ONCE per pod, on the GPU pod itself (container disk, no network volume).
# Clones TokenPowerBench, applies this project's local patch (saves generated
# text for grading), builds a venv with vLLM, and downloads every model
# listed in sweep_config.yaml.
#
# If you kept the same pod running (stopped, not terminated) since the last
# setup, you can skip this and go straight to run_session.sh.
set -euo pipefail

HOME_DIR="${TPB_HOME:-/workspace}"
REPO_URL="${TPB_REPO_URL:-https://github.com/chenxuniu/TokenPowerBench.git}"
REPO_REF="${TPB_REPO_REF:-v1.0.0}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "$HOME_DIR"/{code,models,cache/huggingface,results}

if [ ! -d "$HOME_DIR/code/TokenPowerBench/.git" ]; then
    git clone --branch "$REPO_REF" "$REPO_URL" "$HOME_DIR/code/TokenPowerBench"
else
    git -C "$HOME_DIR/code/TokenPowerBench" fetch --tags
    git -C "$HOME_DIR/code/TokenPowerBench" checkout "$REPO_REF"
fi

# Local patch: api.py now saves batch_*_outputs.json (decoded completion
# text, in request order) for offline grading. See this repo's
# TokenPowerBench/tokenpowerbench/api.py for the diff ([CHANGE] markers).
cp "$SCRIPT_DIR/../TokenPowerBench/tokenpowerbench/api.py" \
   "$HOME_DIR/code/TokenPowerBench/tokenpowerbench/api.py"

python3 -m venv "$HOME_DIR/venv"
# shellcheck disable=SC1091
source "$HOME_DIR/venv/bin/activate"
python -m pip install --upgrade pip
python -m pip install "$HOME_DIR/code/TokenPowerBench[vllm,datasets]"
python -m pip install pyyaml huggingface_hub

mkdir -p "$HOME_DIR/prompts"
cp "$SCRIPT_DIR/../prompts/data"/*.json "$HOME_DIR/prompts/" 2>/dev/null || true

export HF_HOME="$HOME_DIR/cache/huggingface"
python "$SCRIPT_DIR/download_models.py" --config "$SCRIPT_DIR/sweep_config.yaml" --dest "$HOME_DIR/models"

echo "Setup complete under $HOME_DIR. Next: ./run_session.sh"
