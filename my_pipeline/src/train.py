from __future__ import annotations
import os
from datetime import datetime
from typing import Tuple

import tensorflow as tf
from tensorflow.keras import mixed_precision, losses, optimizers, callbacks as keras_callbacks

from .utils import load_config, parse_args, ensure_dir
from .model import build_model

AUTOTUNE = tf.data.AUTOTUNE


def parse_tfrecord(example_proto: tf.Tensor, block_size: int) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    feature_spec = {
        "input_ids": tf.io.FixedLenFeature([block_size], tf.int64),
        "labels": tf.io.FixedLenFeature([block_size], tf.int64),
    }
    example = tf.io.parse_single_example(example_proto, feature_spec)
    input_ids = tf.cast(example["input_ids"], tf.int32)
    labels = tf.cast(example["labels"], tf.int32)
    # Mask: 0 where labels == -100, else 1.0
    mask = tf.cast(tf.not_equal(labels, -100), tf.float32)
    # Replace ignored label values with 0 to keep loss indices valid
    labels = tf.where(tf.equal(labels, -100), tf.zeros_like(labels), labels)
    return input_ids, labels, mask


def dataset_from_tfrecord(path: str, block_size: int, batch_size: int) -> tf.data.Dataset:
    ds = tf.data.TFRecordDataset(path, num_parallel_reads=AUTOTUNE)
    ds = ds.map(lambda x: parse_tfrecord(x, block_size), num_parallel_calls=AUTOTUNE)
    ds = ds.shuffle(1000)
    ds = ds.batch(batch_size, drop_remainder=True)
    # Repackage to (x, y, sample_weight)
    ds = ds.map(lambda x, y, w: (x, y, w), num_parallel_calls=AUTOTUNE)
    ds = ds.prefetch(AUTOTUNE)
    return ds


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    if cfg.get("mixed_precision", False):
        mixed_precision.set_global_policy("mixed_float16")

    train_path = os.path.join(cfg["processed_dir"], "train.tfrecord")
    block_size = int(cfg["block_size"])
    batch_size = int(cfg["batch_size"])

    train_ds = dataset_from_tfrecord(train_path, block_size, batch_size)

    model = build_model(cfg["model_name"])  # T5-like LM
    loss = losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=optimizers.Adam(learning_rate=float(cfg["learning_rate"])), loss=loss)

    # Timestamped run directory
    run_ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    exp = cfg.get("experiment_name", "default")
    run_dir = os.path.join(cfg["log_dir"], exp, run_ts)
    tb_dir = os.path.join(run_dir, "tensorboard")
    ensure_dir(tb_dir)
    callbacks = [keras_callbacks.TensorBoard(log_dir=tb_dir)]

    # If val split exists, use it
    val_path = os.path.join(cfg["processed_dir"], "val.tfrecord")
    val_ds = dataset_from_tfrecord(val_path, block_size, batch_size) if os.path.exists(val_path) else None
    model.fit(train_ds, validation_data=val_ds, epochs=int(cfg["epochs"]), callbacks=callbacks)

    # Save checkpoint in run directory
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    ensure_dir(ckpt_dir)
    model.save_pretrained(ckpt_dir)
    # Write a latest pointer file for convenience
    latest_ptr = os.path.join(cfg["log_dir"], exp, "latest.txt")
    ensure_dir(os.path.dirname(latest_ptr))
    with open(latest_ptr, "w") as f:
        f.write(ckpt_dir)


if __name__ == "__main__":
    main()
