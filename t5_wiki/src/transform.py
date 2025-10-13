from __future__ import annotations
import os
from typing import Dict, Iterable
import numpy as np
import tensorflow as tf
from transformers import T5TokenizerFast

from .utils import load_config, parse_args, ensure_dir



# T5-style span corruption (denoising objective)
import random

def _random_spans_noise_mask(length: int, noise_density: float, mean_span_length: float) -> np.ndarray:
    """Return a boolean mask of shape [length] with random spans masked."""
    num_noise_tokens = int(np.round(length * noise_density))
    num_noise_tokens = min(max(num_noise_tokens, 1), length - 1)
    num_spans = int(np.round(num_noise_tokens / mean_span_length))
    num_spans = max(num_spans, 1)
    # Randomly partition noise tokens into spans
    span_lengths = np.random.multinomial(num_noise_tokens, [1/num_spans]*num_spans)
    # Randomly partition non-noise tokens into gaps
    num_nonnoise = length - num_noise_tokens
    gap_lengths = np.random.multinomial(num_nonnoise, [1/(num_spans+1)]*(num_spans+1))
    mask = np.array([], dtype=bool)
    for gap, span in zip(gap_lengths, np.append(span_lengths, 0)):
        mask = np.concatenate([mask, np.zeros(gap, dtype=bool), np.ones(span, dtype=bool)])
    # Add final gap
    mask = np.concatenate([mask, np.zeros(gap_lengths[-1], dtype=bool)])
    return mask[:length]

def _sentinel_id(tokenizer: T5TokenizerFast, n: int) -> int:
    # T5 uses <extra_id_0>, <extra_id_1>, ... as sentinels
    tok = tokenizer.convert_tokens_to_ids(f"<extra_id_{n}>")
    if isinstance(tok, list):
        return tok[0]
    return tok

def _span_corrupt_block(tokens: np.ndarray, tokenizer: T5TokenizerFast, noise_density=0.15, mean_span_length=3) -> Dict[str, np.ndarray]:
    mask = _random_spans_noise_mask(len(tokens), noise_density, mean_span_length)
    input_ids = []
    target_ids = []
    sentinel_count = 0
    i = 0
    while i < len(tokens):
        if mask[i]:
            # Start of a new masked span
            input_ids.append(_sentinel_id(tokenizer, sentinel_count))
            target_ids.append(_sentinel_id(tokenizer, sentinel_count))
            sentinel_count += 1
            # Copy all masked tokens to target
            while i < len(tokens) and mask[i]:
                target_ids.append(tokens[i])
                i += 1
        else:
            input_ids.append(tokens[i])
            i += 1
    # End sentinel
    target_ids.append(tokenizer.eos_token_id)
    input_ids = np.array(input_ids, dtype=np.int32)
    target_ids = np.array(target_ids, dtype=np.int32)
    # Pad to block_size
    block_size = len(tokens)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    input_ids = np.pad(input_ids, (0, max(0, block_size - len(input_ids))), constant_values=pad_id)[:block_size]
    target_ids = np.pad(target_ids, (0, max(0, block_size - len(target_ids))), constant_values=-100)[:block_size]
    return {"input_ids": input_ids, "labels": target_ids}

def _blocks_from_lines(lines: Iterable[str], tokenizer: T5TokenizerFast, max_length: int, block_size: int, noise_density=0.15, mean_span_length=3) -> Iterable[Dict[str, np.ndarray]]:
    buffer: list[int] = []
    for line in lines:
        ids_list = tokenizer.encode(line.strip(), add_special_tokens=True, truncation=True, max_length=max_length)
        buffer.extend(ids_list)
        while len(buffer) >= block_size:
            block = np.array(buffer[:block_size], dtype=np.int32)
            del buffer[:block_size]
            yield _span_corrupt_block(block, tokenizer, noise_density, mean_span_length)


def _write_records(examples: Iterable[Dict[str, np.ndarray]], out_path: str) -> None:
    ensure_dir(os.path.dirname(out_path))

    def _int64_list_feature(x: np.ndarray) -> tf.train.Feature:
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[int(v) for v in x.tolist()]))

    with tf.io.TFRecordWriter(out_path) as w:
        for ex in examples:
            example = tf.train.Example(features=tf.train.Features(feature={
                "input_ids": _int64_list_feature(ex["input_ids"].astype(np.int64)),
                "labels": _int64_list_feature(ex["labels"].astype(np.int64)),
            }))
            w.write(example.SerializeToString())
    print(f"Wrote TFRecord -> {out_path}")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    raw_path = os.path.join(cfg["raw_dir"], os.path.basename(cfg["raw_data_path"]))
    processed_dir = cfg["processed_dir"]
    ensure_dir(processed_dir)

    tokenizer = T5TokenizerFast.from_pretrained(cfg["model_name"])
    # Deterministic split: every 20th line to val (~5%)
    with open(raw_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    train_lines = (line for i, line in enumerate(lines) if (i % 20) != 0)
    val_lines = (line for i, line in enumerate(lines) if (i % 20) == 0)


    # Use config or defaults for span corruption
    noise_density = float(cfg.get("noise_density", 0.15))
    mean_span_length = float(cfg.get("mean_span_length", 3))
    train_examples = _blocks_from_lines(train_lines, tokenizer, int(cfg["max_length"]), int(cfg["block_size"]), float(noise_density), int(mean_span_length))
    val_examples = _blocks_from_lines(val_lines, tokenizer, int(cfg["max_length"]), int(cfg["block_size"]), float(noise_density), int(mean_span_length))

    _write_records(train_examples, os.path.join(processed_dir, "train.tfrecord"))
    _write_records(val_examples, os.path.join(processed_dir, "val.tfrecord"))


if __name__ == "__main__":
    main()
