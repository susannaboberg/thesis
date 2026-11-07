#measure one  benchmark

import argparse
import json
import os

OUTPUT_TOKENS = {"mmlu": 16, "humaneval": 512}
MAX_MODEL_LEN = 4096

parser = argparse.ArgumentParser()
parser.add_argument("--model-dir", required=True)
parser.add_argument("--task", choices=list(OUTPUT_TOKENS), required=True)
parser.add_argument("--batch-sizes", default="1")
parser.add_argument("--gpu", required=True)
parser.add_argument("--label", required=True)
args = parser.parse_args()

prompts = json.load(open(f"prompts/data/{args.task}_prompts.json"))
answer_keys = json.load(open(f"prompts/data/{args.task}_meta.json"))
results_dir = f"results/{args.gpu}"
os.makedirs(f"{results_dir}/outputs", exist_ok=True)

# write prompt outputs to file, as tokenpowerbench doesn't save outputs natively
def write_outputs(path, records):
    with open(path, "w") as f:
        for key, record in zip(answer_keys, records):
            f.write(json.dumps({"id": key["id"], **record}) + "\n")


# measure energy using tokenpowerbench
def measure():
    import vllm

    from tokenpowerbench import benchmark

    batch_sizes = tuple(int(b) for b in args.batch_sizes.split(","))
    run_key = f"{args.label}__{args.task}__bs{'-'.join(map(str, batch_sizes))}"

    # edited benchmark also saves prompt outputs
    result = benchmark(
        model=args.model_dir,
        prompts=prompts,
        output_tokens=OUTPUT_TOKENS[args.task],
        batch_sizes=batch_sizes,
        phase_profiling=(batch_sizes == (1,)),   # only allowed with serial requests
        monitor="gpu_only",
        output_dir=f"{results_dir}/{run_key}",
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=0.9,
        seed=42,
        temperature=0.0,
    )
    # [TODO]
    for batch in batch_sizes:
        records = json.load(open(result.output_dir / f"batch_{batch}_outputs.json"))
        write_outputs(f"{results_dir}/outputs/{run_key}_bs{batch}.jsonl", records)

    return result

if __name__ == "__main__":
    measure()