#!/usr/bin/env python3
"""
Master automation script for the T5 pipeline.
Runs: ingest, transform, make_test_split, train, evaluate, test, and tune.
"""
import subprocess
import argparse

def run(cmd):
    print(f"\n[RUN] {cmd}")
    subprocess.run(cmd, shell=True, check=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='t5_wiki/configs/default.yaml')
    parser.add_argument('--tune', action='store_true', help='Run hyperparameter tuning')
    parser.add_argument('--test', action='store_true', help='Run test set evaluation')
    parser.add_argument('--skip_test_split', action='store_true', help='Skip test split creation')
    args = parser.parse_args()

    run(f"python3 -m t5_wiki.src.ingest --config {args.config}")
    run(f"python3 -m t5_wiki.src.transform --config {args.config}")
    if not args.skip_test_split:
        run(f"python3 -m t5_wiki.scripts.make_test_split --config {args.config}")
    run(f"python3 -m t5_wiki.src.train --config {args.config}")
    run(f"python3 -m t5_wiki.src.evaluate --config {args.config}")
    if args.test:
        run(f"python3 -m t5_wiki.src.test --config {args.config}")
    if args.tune:
        run(f"python3 t5_wiki/scripts/tune.py")

if __name__ == "__main__":
    main()
