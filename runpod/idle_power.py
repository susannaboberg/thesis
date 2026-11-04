#!/usr/bin/env python3
"""Sample idle GPU power for a fixed window before loading any model.

Uses TokenPowerBench's own GPU monitor so the output has the same shape as
batch_*_power_samples.json from a real run, for an easy apples-to-apples
comparison in graphing.ipynb.

Usage:
    python idle_power.py --seconds 60 --out results/session_.../idle_power.json
"""
import argparse
import json
import time

from tokenpowerbench.energy import create_monitor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=int, default=60)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    monitor = create_monitor("gpu_only")
    try:
        monitor.start()
        time.sleep(args.seconds)
        monitor.stop()
        with open(args.out, "w") as f:
            json.dump(monitor.samples, f, indent=2)
        print(f"Idle power samples written to {args.out}")
    finally:
        monitor.close()


if __name__ == "__main__":
    main()
