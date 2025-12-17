#!/usr/bin/env python3
"""
Run the complete t5_xsum pipeline.
"""
import subprocess
import sys
from pathlib import Path
import argparse

def run(cmd):
    print(f"🚀 Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ Command failed: {cmd}")
        print(f"Error: {result.stderr}")
        sys.exit(1)
    print(f"✅ Completed: {cmd}")

def main():
    parser = argparse.ArgumentParser(description="Run t5_xsum pipeline")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    
    steps = [
        f"python3 -m t5_xsum.src.ingest --config {args.config}",
        f"python3 -m t5_xsum.scripts.make_test_split_pt --config {args.config} --num_lines 1000",
        f"python3 -m t5_xsum.src.train_pt --config {args.config}",
        f"python3 -m t5_xsum.src.advanced_eval_pt --config {args.config}",
        f"python3 -m t5_xsum.src.test_pt --config {args.config}",
        f"python3 -m t5_xsum.src.export_pt --config {args.config}"
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
    
    print("\n🎉 Pipeline completed successfully!")

if __name__ == "__main__":
    main()
