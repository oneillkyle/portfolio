#!/usr/bin/env python3
"""
Run all steps of the PyTorch t5_wiki pipeline.
Runs: ingest, create test split, train, evaluate, test, and export.
"""
import subprocess
import sys
import argparse

def run(cmd):
    """Run a shell command, exit on failure."""
    print(f"🚀 Running: {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"❌ Command failed with exit code {result.returncode}: {cmd}")
        sys.exit(result.returncode)
    print(f"✅ Command completed: {cmd}")

def main():
    parser = argparse.ArgumentParser(description='Run all t5_wiki pipeline steps')
    parser.add_argument('--config', type=str, default='t5_wiki/configs/default.yaml', help='Path to config file')
    args = parser.parse_args()

    steps = [
        f"python3 -m t5_wiki.src.ingest --config {args.config}",
        f"python3 -m t5_wiki.scripts.make_test_split_pt --config {args.config} --num_lines 1000",
        f"python3 -m t5_wiki.src.train_pt --config {args.config}",
        f"python3 -m t5_wiki.src.advanced_eval_pt --config {args.config}",
        f"python3 -m t5_wiki.src.test_pt --config {args.config}",
        f"python3 -m t5_wiki.src.export_pt --config {args.config}"
    ]
    
    for i, step in enumerate(steps, 1):
        print(f"\n[{i}/{len(steps)}] {step.split()[2].split('.')[-1].upper()}")
        try:
            run(step)
        except SystemExit:
            if i <= 3:  # Critical steps: ingest, test split, train
                raise
            else:  # Optional steps: evaluate, test, export
                print(f"⚠️  Step {i} failed but continuing...")
    
    print("\n🎉 Pipeline completed!")

if __name__ == "__main__":
    main()
