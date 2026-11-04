"""build prompts for MMLU and HumanEval -- raw text is needed for TokenPowerBench

    pip install datasets transformers
    python prepare_prompts.py

    data/{task}_prompts.json (list of strings), data/{task}_meta.json (answers/tests for grading)
"""
import json
import os
import random

from datasets import load_dataset
from transformers import AutoTokenizer

NUM_MMLU = 200 # using all 164 problems for HumanEval
random.seed(0)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
os.makedirs("data", exist_ok=True)


def format_as_chat(content):
    return tokenizer.apply_chat_template([{"role": "user", "content": content}],
                                   tokenize=False, add_generation_prompt=True)


def save_prompt_files(task, prompts, answer_keys):
    json.dump(prompts, open(f"data/{task}_prompts.json", "w"), indent=1)
    json.dump(answer_keys, open(f"data/{task}_meta.json", "w"), indent=1)
    print(f"{task}: {len(prompts)} prompts") # print number prompts after saving


# MMLU
mmlu_test_set = load_dataset("cais/mmlu", "all", split="test")
sampled_row_nums = random.sample(range(len(mmlu_test_set)), NUM_MMLU) # list sample 200 prompt row numbers from mmlu
prompts, answer_keys = [], []
for i in sampled_row_nums:
    question_row = mmlu_test_set[i]
    choices = "\n".join(f"{letter}. {choice}" for letter, choice in zip("ABCD", question_row["choices"]))
    prompts.append(format_as_chat(f"{question_row['question']}\n\n{choices}\n\n"
                        "Answer with only the letter of the correct choice."))
    answer_keys.append({
        "id": f"mmlu-{i}", 
        "answer": "ABCD"[question_row["answer"]],
        "subject": question_row["subject"]})
# answer stored as 0 to 3
save_prompt_files("mmlu", prompts, answer_keys)


# HumanEval
humaneval_test_set = load_dataset("openai/openai_humaneval", split="test")
prompts, answer_keys = [], []
for problem in humaneval_test_set:
    prompts.append(format_as_chat("Complete the following Python function. Return the full function "
                        f"in a single Python code block and nothing else.\n\n```python\n{problem['prompt']}```"))
    answer_keys.append({
        "id": problem["task_id"], 
        "code_prompt": problem["prompt"],
        "test": problem["test"], 
        "entry_point": problem["entry_point"]
    })
save_prompt_files("humaneval", prompts, answer_keys)