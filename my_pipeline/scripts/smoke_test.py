#!/usr/bin/env python3
from __future__ import annotations
import os
import tempfile
from typing import Iterable, Dict

import numpy as np
import tensorflow as tf
from transformers import T5TokenizerFast

from my_pipeline.src.utils import load_config, ensure_dir
from my_pipeline.src.model import build_model
from my_pipeline.src.train import dataset_from_tfrecord


def write_simple_tfrecord(lines: Iterable[str], tokenizer: T5TokenizerFast, block_size: int, out_path: str) -> None:
    ensure_dir(os.path.dirname(out_path))

    def _int64_list_feature(x: np.ndarray) -> tf.train.Feature:
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[int(v) for v in x.tolist()]))

    buf: list[int] = []
    with tf.io.TFRecordWriter(out_path) as w:
        for line in lines:
            ids = tokenizer.encode(line.strip(), add_special_tokens=True, truncation=True, max_length=block_size)
            buf.extend(ids)
            while len(buf) >= block_size:
                block = np.array(buf[:block_size], dtype=np.int32)
                del buf[:block_size]
                labels = np.concatenate([block[1:], np.array([-100], dtype=np.int32)])
                pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
                labels = np.where(block == pad_id, -100, labels).astype(np.int32)
                ex = tf.train.Example(features=tf.train.Features(feature={
                    "input_ids": _int64_list_feature(block.astype(np.int64)),
                    "labels": _int64_list_feature(labels.astype(np.int64)),
                }))
                w.write(ex.SerializeToString())


def main():
    cfg = load_config("my_pipeline/configs/default.yaml")
    raw_path = cfg["raw_data_path"]
    assert os.path.exists(raw_path), f"Missing raw file: {raw_path}"

    # Take a tiny subset
    with open(raw_path, "r", encoding="utf-8") as f:
        lines = [next(f) for _ in range(200)]

    tokenizer = T5TokenizerFast.from_pretrained(cfg["model_name"])

    tmp_dir = os.path.join("my_pipeline", "data", "processed", "smoke")
    ensure_dir(tmp_dir)
    tfrec = os.path.join(tmp_dir, "smoke.tfrecord")
    write_simple_tfrecord(lines, tokenizer, int(cfg["block_size"]), tfrec)

    ds = dataset_from_tfrecord(tfrec, int(cfg["block_size"]), batch_size=2)

    model = build_model(cfg["model_name"]) 
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss=loss)
    # Train one step
    model.fit(ds.take(1), epochs=1)
    print("Smoke test completed.")


if __name__ == "__main__":
    main()
