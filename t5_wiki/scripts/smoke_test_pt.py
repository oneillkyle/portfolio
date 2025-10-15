#!/usr/bin/env python3
"""
PyTorch smoke test: runs a tiny end-to-end train/eval on a small subset.
"""
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments
from datasets import Dataset
import os
from t5_wiki.src.utils import load_config, ensure_dir

def main():
    cfg = load_config("t5_wiki/configs/default.yaml")

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
        raise FileNotFoundError(f"Could not locate raw_data_path for smoke test. Tried: {candidates}")

    # Take a tiny subset
    with open(raw_path, "r", encoding="utf-8") as f:
        lines = [next(f) for _ in range(32)]

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])

    def preprocess(example):
        ids = tokenizer.encode(example["text"], truncation=True, max_length=int(cfg["block_size"]))
        return {"input_ids": ids, "labels": ids}

    dataset = Dataset.from_dict({"text": lines})
    dataset = dataset.map(preprocess, remove_columns=["text"])
    split = dataset.train_test_split(test_size=0.2, seed=42)
    train_ds, val_ds = split["train"], split["test"]

    model = AutoModelForSeq2SeqLM.from_pretrained(cfg["model_name"])
    training_args = TrainingArguments(
        output_dir="t5_wiki/logs/smoke_pt",
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        num_train_epochs=1,
        evaluation_strategy="epoch",
        logging_dir="t5_wiki/logs/smoke_pt/tensorboard",
        report_to=["tensorboard"],
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
    )
    trainer.train()
    print("PyTorch smoke test completed.")

if __name__ == "__main__":
    main()
