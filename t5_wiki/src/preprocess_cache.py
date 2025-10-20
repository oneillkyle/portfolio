#!/usr/bin/env python3
"""
Preprocess raw wiki corpus into tokenized, span-corrupted Arrow dataset cached on disk.
- Reads the canonical raw_data_path from config
- Tokenizes and applies T5 span corruption
- Saves to disk (t5_wiki/data/tokenized by default)
"""
from __future__ import annotations
import os
import argparse
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
from .utils import load_config, ensure_dir


def _random_spans_noise_mask(length: int, noise_density: float, mean_span_length: float) -> np.ndarray:
    num_noise_tokens = int(np.round(length * noise_density))
    num_noise_tokens = min(max(num_noise_tokens, 1), max(length - 1, 1))
    num_spans = int(np.round(num_noise_tokens / max(mean_span_length, 1)))
    num_spans = max(num_spans, 1)
    span_lengths = np.random.multinomial(num_noise_tokens, [1/num_spans]*num_spans)
    num_nonnoise = max(length - num_noise_tokens, 0)
    gap_lengths = np.random.multinomial(num_nonnoise, [1/(num_spans+1)]*(num_spans+1))
    mask = np.array([], dtype=bool)
    for gap, span in zip(gap_lengths, np.append(span_lengths, 0)):
        mask = np.concatenate([mask, np.zeros(gap, dtype=bool), np.ones(span, dtype=bool)])
    mask = np.concatenate([mask, np.zeros(gap_lengths[-1], dtype=bool)])
    return mask[:length]


def _sentinel_id(tokenizer, n: int) -> int:
    tok = tokenizer.convert_tokens_to_ids(f"<extra_id_{n}>")
    if isinstance(tok, list):
        return tok[0]
    return tok


def _get_args():
    p = argparse.ArgumentParser(description="Preprocess and cache tokenized dataset")
    p.add_argument("--config", type=str, default="t5_wiki/configs/default.yaml")
    p.add_argument("--force", action="store_true", help="Rebuild cache even if it already exists")
    return p.parse_args()


def main():
    args = _get_args()
    cfg = load_config(args.config)

    raw_path = cfg["raw_data_path"]
    tokenized_dir = cfg.get("tokenized_dir", "t5_wiki/data/tokenized")
    # Skip if cache exists unless forced
    if os.path.exists(tokenized_dir) and not args.force:
        print(f"✅ Tokenized cache already exists at {tokenized_dir}. Use --force to rebuild.")
        return
    ensure_dir(tokenized_dir)

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])

    # Load raw text dataset (non-streaming) so we can map in parallel and save
    ds = load_dataset("text", data_files={"train": raw_path})["train"]

    noise_density = float(cfg.get("noise_density", 0.15))
    mean_span_length = float(cfg.get("mean_span_length", 3))
    block_size = int(cfg.get("block_size", 256))

    def preprocess(batch):
        input_ids_list = []
        labels_list = []
        for txt in batch["text"]:
            ids = tokenizer.encode(txt, truncation=True, max_length=block_size)
            if len(ids) == 0:
                # Skip empty
                input_ids_list.append([tokenizer.pad_token_id] * block_size)
                labels_list.append([-100] * block_size)
                continue
            mask = _random_spans_noise_mask(len(ids), noise_density, mean_span_length)
            input_ids = []
            target_ids = []
            sentinel_count = 0
            i = 0
            while i < len(ids):
                if mask[i]:
                    input_ids.append(_sentinel_id(tokenizer, sentinel_count))
                    target_ids.append(_sentinel_id(tokenizer, sentinel_count))
                    sentinel_count += 1
                    while i < len(ids) and mask[i]:
                        target_ids.append(ids[i])
                        i += 1
                else:
                    input_ids.append(ids[i])
                    i += 1
            target_ids.append(tokenizer.eos_token_id)
            pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
            input_ids = np.pad(input_ids, (0, max(0, block_size - len(input_ids))), constant_values=pad_id)[:block_size]
            target_ids = np.pad(target_ids, (0, max(0, block_size - len(target_ids))), constant_values=-100)[:block_size]
            input_ids_list.append(input_ids.tolist())
            labels_list.append(target_ids.tolist())
        return {"input_ids": input_ids_list, "labels": labels_list}

    num_proc = int(cfg.get("num_proc", 1))
    map_kwargs = {"batched": True, "batch_size": 100, "remove_columns": ["text"]}
    try:
        ds = ds.map(preprocess, num_proc=num_proc, **map_kwargs)
    except TypeError:
        # Older datasets version may not support num_proc; fall back to single-process
        ds = ds.map(preprocess, **map_kwargs)

    ds.save_to_disk(tokenized_dir)
    print(f"✅ Saved tokenized dataset to {tokenized_dir}")


if __name__ == "__main__":
    main()
