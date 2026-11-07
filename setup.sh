set -euo pipefail
GPU=$1

python -m venv /workspace/venv
source /workspace/venv/bin/activate
pip install --upgrade pip

# TokenPowerBench pinned to the commit our patch targets (not v1.0.0 -- that
# tag predates the engine base-class refactor our patch applies to), with the
# write-outputs patch applied (adds batch_*_outputs.json so we can grade
# MMLU/HumanEval correctness, not just power/throughput).
git clone --quiet https://github.com/chenxuniu/TokenPowerBench.git /tmp/TokenPowerBench
git -C /tmp/TokenPowerBench checkout --quiet 9a50272
git -C /tmp/TokenPowerBench apply "$(pwd)/patches/write-outputs.patch"
pip install "/tmp/TokenPowerBench[vllm,datasets]"

MODELS=(Qwen2.5-0.5B-Instruct Qwen2.5-3B-Instruct Qwen2.5-7B-Instruct)
mkdir -p models
for model in "${MODELS[@]}"; do
    huggingface-cli download "Qwen/$model" --local-dir "models/$model"
done

mkdir -p "results/$GPU"
nvidia-smi -q > "results/$GPU/nvidia_smi_q.txt"
pip freeze > "results/$GPU/pip_freeze.txt"