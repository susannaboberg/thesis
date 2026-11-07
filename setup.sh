set -euo pipefail
GPU=$1

python -m venv /workspace/venv
source /workspace/venv/bin/activate
pip install --upgrade pip

# TokenPowerBench w/ patches to save prompt outputs
git clone --quiet https://github.com/chenxuniu/TokenPowerBench.git /tmp/TokenPowerBench
git -C /tmp/TokenPowerBench checkout --quiet 9a50272
git -C /tmp/TokenPowerBench apply "$(pwd)/patches/write-outputs.patch"
pip install "/tmp/TokenPowerBench[vllm,datasets]"

MODELS=(Qwen2.5-0.5B-Instruct Qwen2.5-3B-Instruct Qwen2.5-7B-Instruct)
mkdir -p models
for model in "${MODELS[@]}"; do
    python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/$model', local_dir='models/$model')"
done

mkdir -p "results/$GPU"
nvidia-smi -q > "results/$GPU/nvidia_smi_q.txt"
pip freeze > "results/$GPU/pip_freeze.txt"