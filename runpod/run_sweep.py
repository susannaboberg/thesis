#!/usr/bin/env python3
"""Config-driven benchmark sweep over models x prompt sources.

For each (model, prompt_source) pair, calls tokenpowerbench.api.benchmark()
once with the full batch_sizes list, so the model loads exactly once per
pair (TokenPowerBench itself loops over batch sizes inside one engine load -
see tokenpowerbench/api.py's `for batch in options["batch_sizes"]`).

A completed pair is marked with a small JSON file under <output_dir>/.done/,
so re-running this script after a crash or a new pod skips finished work
instead of redoing it. A failure (e.g. OOM on a model too large for this
GPU) is logged and the sweep moves on to the next pair.

Usage:
    python run_sweep.py --config sweep_config.yaml --home /workspace
"""
import argparse
import json
import time
import traceback
from pathlib import Path

import yaml

from tokenpowerbench import api


def load_config(path):
    return yaml.safe_load(Path(path).read_text())


def resolve_source_kwargs(source, home):
    if "prompts_file" in source:
        return {"prompts_file": home / "prompts" / Path(source["prompts_file"]).name}
    return {"dataset": source["dataset"], "num_samples": source.get("num_samples")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path(__file__).parent / "sweep_config.yaml")
    parser.add_argument("--home", type=Path, default=Path("/workspace"))
    args = parser.parse_args()

    config = load_config(args.config)
    home = args.home
    output_root = home / config["output_dir"]
    done_dir = output_root / ".done"
    done_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_root / "sweep_log.jsonl"

    pairs = [(model, source) for model in config["models"] for source in config["prompt_sources"]]
    print(f"Sweep: {len(pairs)} (model, prompt_source) pairs")

    for index, (model, source) in enumerate(pairs):
        run_name = f"{model['local_name']}__{source['name']}"
        marker = done_dir / f"{run_name}.json"
        if marker.is_file():
            print(f"[{index + 1}/{len(pairs)}] {run_name}: already done, skipping")
            continue

        print(f"[{index + 1}/{len(pairs)}] {run_name}: running")
        entry = {"run_name": run_name, "model": model["local_name"], "prompt_source": source["name"],
                 "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        try:
            result = api.benchmark(
                model=str(home / "models" / model["local_name"]),
                batch_sizes=config["batch_sizes"],
                output_tokens=config["output_tokens"],
                max_model_len=config.get("max_model_len"),
                gpu_memory_utilization=config.get("gpu_memory_utilization", 0.9),
                monitor=config.get("monitor", "gpu_only"),
                seed=config.get("seed", 42),
                temperature=config.get("temperature", 0.0),
                output_dir=output_root / run_name,
                **resolve_source_kwargs(source, home),
            )
            entry.update(status="completed", run_dir=str(result.output_dir))
            marker.write_text(json.dumps(entry, indent=2))
            print(f"[{index + 1}/{len(pairs)}] {run_name}: completed -> {result.output_dir}")
        except Exception as exc:
            entry.update(status="failed", error=str(exc), traceback=traceback.format_exc())
            print(f"[{index + 1}/{len(pairs)}] {run_name}: FAILED: {exc}")

        with open(log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

        is_last = index == len(pairs) - 1
        pause = config.get("pause_between_runs_s", 0)
        if not is_last and pause:
            print(f"Pausing {pause}s before the next run (GPU cooldown)")
            time.sleep(pause)

    print(f"Sweep finished. Log: {log_path}")


if __name__ == "__main__":
    main()
