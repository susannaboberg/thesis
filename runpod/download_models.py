#!/usr/bin/env python3
"""Download every model listed in sweep_config.yaml to a local directory.

Usage:
    python download_models.py --config sweep_config.yaml --dest /workspace/models

Run this once per pod (or skip it on a later pod if you kept the same
container disk). Prints the resolved commit hash for any model pinned to
"main" so you can copy it back into sweep_config.yaml for a reproducible run.
"""
import argparse
from pathlib import Path

import yaml
from huggingface_hub import HfApi, snapshot_download

ALLOW_PATTERNS = ["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path(__file__).parent / "sweep_config.yaml")
    parser.add_argument("--dest", type=Path, required=True)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())
    api = HfApi()
    args.dest.mkdir(parents=True, exist_ok=True)

    for model in config["models"]:
        repo_id, revision, local_name = model["repo_id"], model["revision"], model["local_name"]
        resolved = api.model_info(repo_id, revision=revision).sha
        if revision == "main":
            print(f"{repo_id}: \"main\" resolves to {resolved} -- "
                  f"pin this in sweep_config.yaml for a reproducible run")
        local_dir = args.dest / local_name
        if (local_dir / "config.json").is_file():
            print(f"{local_name}: already present at {local_dir}, skipping")
            continue
        print(f"Downloading {repo_id}@{revision} -> {local_dir}")
        snapshot_download(
            repo_id=repo_id,
            revision=revision,
            local_dir=str(local_dir),
            allow_patterns=ALLOW_PATTERNS,
        )
        (local_dir / "RESOLVED_REVISION.txt").write_text(f"{revision} -> {resolved}\n")


if __name__ == "__main__":
    main()
