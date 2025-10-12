from __future__ import annotations
import os
import math
import tensorflow as tf
from tensorflow.keras import losses, metrics

from .utils import load_config, parse_args
from .train import dataset_from_tfrecord
from .model import build_model


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    # Prefer validation split if present
    val_path = os.path.join(cfg["processed_dir"], "val.tfrecord")
    eval_path = val_path if os.path.exists(val_path) else os.path.join(cfg["processed_dir"], "train.tfrecord")
    if not os.path.exists(eval_path):
        print("No eval TFRecord found; skipping evaluation.")
        return

    ds = dataset_from_tfrecord(eval_path, int(cfg["block_size"]), int(cfg["batch_size"]))
    # Load latest checkpoint if exists
    exp = cfg.get("experiment_name", "default")
    latest_ptr = os.path.join(cfg["log_dir"], exp, "latest.txt")
    model_src = open(latest_ptr).read().strip() if os.path.exists(latest_ptr) else cfg["model_name"]
    model = build_model(model_src)

    # Evaluate loss
    loss_fn = losses.SparseCategoricalCrossentropy(from_logits=True)
    avg_loss = metrics.Mean()

    for batch in ds.take(100):  # limit for speed
        # batch may be (x,y,w) or (x,y) depending on builder
        if isinstance(batch, tuple) and len(batch) == 3:
            x, y, w = batch
            logits = model(x, training=False)
            loss = loss_fn(y, logits, sample_weight=w)
        else:
            x, y = batch
            logits = model(x, training=False)
            loss = loss_fn(y, logits)
        avg_loss.update_state(loss)

    loss_val = float(avg_loss.result().numpy())
    ppl = math.exp(loss_val)
    print({"loss": loss_val, "perplexity": ppl})


if __name__ == "__main__":
    main()
