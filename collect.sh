# tmux? from repo folder rtx 4090

set -u 
GPU=$1
export CUDA_VISIBLE_DEVICES=0
mkdir -p "results/$GPU"

MODELS=(
  "qwen2.5-0.5b|models/Qwen2.5-0.5B-Instruct"
  "qwen2.5-3b|models/Qwen2.5-3B-Instruct"
  "qwen2.5-7b|models/Qwen2.5-7B-Instruct"
)

TASKS=(mmlu humaneval)
BATCH_GROUPS=("1" "8,32,128")

# loop through models, tasks (humaneval & mmlu), batch sizes (for single GPU)
for model in "${MODELS[@]}"; do
  IFS='|' read -r label model_dir <<< "$model"
  for task in "${TASKS[@]}"; do
    for batch_group in "${BATCH_GROUPS[@]}"; do
      python single_run.py --label "$label" --model-dir "$model_dir" --task "$task" --batch-sizes "$batch_group" --gpu "$GPU" \
      || echo "FAILED measure single_run.py $label $task $batch_group" >> "results/$GPU/failed.txt"
      sleep 20
    done
  done
done

echo "complete"