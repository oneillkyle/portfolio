#!/usr/bin/env python3
"""
Create a test split from raw data for PyTorch pipeline.
"""
import os
import argparse
from t5_wiki.src.utils import load_config, ensure_dir

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='t5_wiki/configs/default.yaml')
    parser.add_argument('--num_lines', type=int, default=1000, help='Number of lines to use for test set')
    parser.add_argument('--start_line', type=int, default=0, help='Starting line for test set')
    args = parser.parse_args()

    cfg = load_config(args.config)
    # Robustly resolve the raw data path
    raw_cfg = cfg["raw_data_path"]
    candidates = []
    if os.path.isabs(raw_cfg):
        candidates.append(raw_cfg)
    else:
        candidates.append(os.path.abspath(raw_cfg))
    candidates.append(os.path.join(cfg.get("raw_dir", ""), os.path.basename(raw_cfg)))
    raw_path = None
    for p in candidates:
        if p and os.path.exists(p):
            raw_path = p
            break
    if raw_path is None:
        raise FileNotFoundError(f"Could not locate raw_data_path. Tried: {candidates}. Run ingest first: python -m t5_wiki.src.ingest --config {args.config}")
    processed_dir = cfg["processed_dir"]
    ensure_dir(processed_dir)

    print(f"Creating test split from {raw_path}")
    print(f"Lines: {args.start_line} to {args.start_line + args.num_lines}")
    
    # Read specific lines for test set
    test_lines = []
    with open(raw_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= args.start_line and i < args.start_line + args.num_lines:
                test_lines.append(line.strip())
            elif i >= args.start_line + args.num_lines:
                break
    
    # Write test.txt
    test_path = os.path.join(processed_dir, "test.txt")
    with open(test_path, "w", encoding="utf-8") as f:
        for line in test_lines:
            if line:  # Skip empty lines
                f.write(line + "\n")
    
    print(f"✅ Created test split: {test_path}")
    print(f"📊 Test samples: {len(test_lines)}")

if __name__ == "__main__":
    main()