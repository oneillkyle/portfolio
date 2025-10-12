from __future__ import annotations
import os
from typing import Dict, Iterable
import numpy as np
import tensorflow as tf
from transformers import T5TokenizerFast

from .utils import load_config, parse_args, ensure_dir


def _blocks_from_lines(lines: Iterable[str], tokenizer: T5TokenizerFast, max_length: int, block_size: int) -> Iterable[Dict[str, np.ndarray]]:
    buffer: list[int] = []
    for line in lines:
        # tokenize one line to ids (no per-line padding to avoid type issues); limit length
        ids_list = tokenizer.encode(line.strip(), add_special_tokens=True, truncation=True, max_length=max_length)
        buffer.extend(ids_list)
        # emit any full blocks
        while len(buffer) >= block_size:
            block = np.array(buffer[:block_size], dtype=np.int32)
            del buffer[:block_size]
            labels = np.concatenate([block[1:], np.array([-100], dtype=np.int32)])
            # mask pads
            pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
            labels = np.where(block == pad_id, -100, labels).astype(np.int32)
            yield {"input_ids": block, "labels": labels}


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

    train_examples = _blocks_from_lines(train_lines, tokenizer, int(cfg["max_length"]), int(cfg["block_size"]))
    val_examples = _blocks_from_lines(val_lines, tokenizer, int(cfg["max_length"]), int(cfg["block_size"]))

    _write_records(train_examples, os.path.join(processed_dir, "train.tfrecord"))
    _write_records(val_examples, os.path.join(processed_dir, "val.tfrecord"))


if __name__ == "__main__":
    main()
