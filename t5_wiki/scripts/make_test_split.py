#!/usr/bin/env python3
"""
Create a test TFRecord from a portion of the raw data, using the same preprocessing as transform.py.
"""
import os
import argparse
from transformers import T5TokenizerFast
from t5_wiki.src.utils import load_config, ensure_dir
from t5_wiki.src.transform import _blocks_from_lines, _write_records

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='t5_wiki/configs/default.yaml')
    parser.add_argument('--num_lines', type=int, default=1000, help='Number of lines to use for test set')
    args = parser.parse_args()

    cfg = load_config(args.config)
    raw_path = os.path.join(cfg["raw_dir"], os.path.basename(cfg["raw_data_path"]))
    processed_dir = cfg["processed_dir"]
    ensure_dir(processed_dir)

    tokenizer = T5TokenizerFast.from_pretrained(cfg["model_name"])
    with open(raw_path, "r", encoding="utf-8") as f:
        lines = [next(f) for _ in range(args.num_lines)]

    noise_density = float(cfg.get("noise_density", 0.15))
    mean_span_length = int(cfg.get("mean_span_length", 3))
    max_length = int(cfg["max_length"])
    block_size = int(cfg["block_size"])

    test_examples = _blocks_from_lines(lines, tokenizer, max_length, block_size, noise_density, mean_span_length)
    out_path = os.path.join(processed_dir, "test.tfrecord")
    _write_records(test_examples, out_path)
    print(f"Wrote test TFRecord -> {out_path}")

if __name__ == "__main__":
    main()
