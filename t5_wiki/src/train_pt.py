from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments
from datasets import load_dataset, Dataset
import numpy as np
import os
import yaml
from .utils import load_config, parse_args, ensure_dir

import numpy as np
def _random_spans_noise_mask(length: int, noise_density: float, mean_span_length: float) -> np.ndarray:
    num_noise_tokens = int(np.round(length * noise_density))
    num_noise_tokens = min(max(num_noise_tokens, 1), length - 1)
    num_spans = int(np.round(num_noise_tokens / mean_span_length))
    num_spans = max(num_spans, 1)
    span_lengths = np.random.multinomial(num_noise_tokens, [1/num_spans]*num_spans)
    num_nonnoise = length - num_noise_tokens
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

def span_corrupt_example(example, tokenizer, noise_density=0.15, mean_span_length=3, block_size=256):
    ids = tokenizer.encode(example["text"], truncation=True, max_length=block_size)
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
    return {"input_ids": input_ids.tolist(), "labels": target_ids.tolist()}

def main():
    args = parse_args()
    cfg = load_config(args.config)
    model_name = cfg["model_name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load raw data (assume one line per example)
    raw_path = os.path.join(cfg["raw_dir"], os.path.basename(cfg["raw_data_path"]))
    with open(raw_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    dataset = Dataset.from_dict({"text": lines})

    # Apply span corruption
    dataset = dataset.map(lambda ex: span_corrupt_example(ex, tokenizer,
        noise_density=float(cfg.get("noise_density", 0.15)),
        mean_span_length=int(cfg.get("mean_span_length", 3)),
        block_size=int(cfg["block_size"])),
        remove_columns=["text"])

    # Split train/val
    split = dataset.train_test_split(test_size=0.05, seed=cfg.get("random_seed", 42))
    train_ds, val_ds = split["train"], split["test"]

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    training_args = TrainingArguments(
        output_dir=cfg["log_dir"],
        per_device_train_batch_size=int(cfg["batch_size"]),
        per_device_eval_batch_size=int(cfg["batch_size"]),
        num_train_epochs=int(cfg["epochs"]),
        learning_rate=float(cfg["learning_rate"]),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir=os.path.join(cfg["log_dir"], "tensorboard"),
        report_to=["tensorboard"],
        fp16=cfg.get("mixed_precision", False),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
    )
    trainer.train()
    trainer.save_model(cfg["log_dir"])

if __name__ == "__main__":
    main()
