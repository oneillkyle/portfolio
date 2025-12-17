from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from datasets import load_dataset, Dataset, load_from_disk
import numpy as np
import os
import yaml
from .utils import load_config, parse_args, ensure_dir
import torch

print("CUDA available:", torch.cuda.is_available())
print("Device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("Device name:", torch.cuda.get_device_name(0))

import transformers
print(transformers.__file__)
from transformers import TrainingArguments
print(TrainingArguments.__init__.__doc__)

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

    print("[LOG] Starting data load...")
    # Resolve raw data file path robustly:
    # 1) Use cfg["raw_data_path"] directly if it exists (absolute or relative)
    # 2) Fallback to raw_dir + basename(raw_data_path)
    raw_data_path = cfg["raw_data_path"]
    candidate_paths = []
    if os.path.isabs(raw_data_path):
        candidate_paths.append(raw_data_path)
    else:
        candidate_paths.append(os.path.abspath(raw_data_path))
    # Fallback location inside configured raw_dir
    candidate_paths.append(os.path.join(cfg.get("raw_dir", ""), os.path.basename(raw_data_path)))

    raw_path = None
    for p in candidate_paths:
        if p and os.path.exists(p):
            raw_path = p
            break
    if raw_path is None:
        raise FileNotFoundError(f"Could not locate raw_data_path. Tried: {candidate_paths}")
    print(f"[LOG] Using raw data file: {raw_path}")

    # Try cached tokenized dataset first for maximum throughput
    tokenized_dir = cfg.get("tokenized_dir", "t5_wiki/data/tokenized")
    ds_cached = None
    if os.path.exists(tokenized_dir):
        try:
            ds_cached = load_from_disk(tokenized_dir)
            print(f"[LOG] Loaded cached dataset from {tokenized_dir}")
        except Exception as e:
            print(f"[LOG] Failed to load cached dataset: {e}")

    if ds_cached is not None:
        # Non-streaming, already tokenized
        full_ds = ds_cached
        is_streaming = False
    else:
        # Fallback: streaming raw text (slower)
        print("[LOG] Loading dataset in streaming mode...")
        full_ds = load_dataset(
            "text",
            data_files={"train": raw_path},
            split="train",
            streaming=True
        )
        is_streaming = True
        print("[LOG] Streaming dataset loaded.")

    # Map preprocessing (tokenization, span corruption)
    def preprocess(example):
        ids = tokenizer.encode(example["text"], truncation=True, max_length=int(cfg["block_size"]))
        mask = _random_spans_noise_mask(len(ids), float(cfg.get("noise_density", 0.15)), int(cfg.get("mean_span_length", 3)))
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
        input_ids = np.pad(input_ids, (0, max(0, int(cfg["block_size"]) - len(input_ids))), constant_values=pad_id)[:int(cfg["block_size"])]
        target_ids = np.pad(target_ids, (0, max(0, int(cfg["block_size"]) - len(target_ids))), constant_values=-100)[:int(cfg["block_size"])]
        return {"input_ids": input_ids.tolist(), "labels": target_ids.tolist()}

    print("[LOG] Applying preprocessing...")
    if is_streaming:
        dataset = full_ds.map(preprocess)
    else:
        # Already tokenized
        dataset = full_ds
    print("[LOG] Preprocessing complete.")

    # Split train/val using islice for streaming datasets
    from itertools import islice
    val_size = int(cfg.get("val_size", 1000))
    if is_streaming:
        val_ds = dataset.take(val_size)
        # Shuffle the streaming training dataset with a buffer for better mixing
        train_ds = dataset.skip(val_size).shuffle(buffer_size=10000, seed=int(cfg.get("random_seed", 42)))
    else:
        # Non-streaming random split
        split = dataset.train_test_split(test_size=val_size, seed=int(cfg.get("random_seed", 42)))
        train_ds, val_ds = split["train"], split["test"]

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # For streaming datasets, set max_steps instead of num_train_epochs
    max_steps = int(cfg.get("max_steps", 10000))  # Read from config or default to 10,000
    step_interval = int(cfg.get("save_interval", 1000))
    logging_steps = int(cfg.get("logging_steps", max(50, step_interval // 10)))

    training_args = TrainingArguments(
        output_dir=cfg["log_dir"],
        per_device_train_batch_size=int(cfg["batch_size"]),
        per_device_eval_batch_size=int(cfg.get("per_device_eval_batch_size", cfg["batch_size"])),
        gradient_accumulation_steps=int(cfg.get("gradient_accumulation_steps", 1)),
        eval_accumulation_steps=int(cfg.get("eval_accumulation_steps", 1)),
        max_steps=max_steps,
        learning_rate=float(cfg["learning_rate"]),
        eval_strategy="steps",  # Use steps for progress
        save_strategy="steps",
        eval_steps=step_interval,
        save_steps=step_interval,
        logging_steps=logging_steps,
        logging_dir=os.path.join(cfg["log_dir"], "tensorboard"),
        report_to=["tensorboard", "wandb"] if cfg.get("use_wandb", False) else ["tensorboard"],
        fp16=cfg.get("mixed_precision", False),
        run_name=cfg.get("run_name", "t5-wiki-training"),
        seed=int(cfg.get("random_seed", 42)),
        data_seed=int(cfg.get("random_seed", 42)),
        save_total_limit=int(cfg.get("save_total_limit", 3)),
        dataloader_num_workers=int(cfg.get("dataloader_num_workers", 0)),
        dataloader_pin_memory=True,
        # Speed optimizations
        gradient_checkpointing=True,  # Save memory at slight speed cost
        optim="adamw_torch_fused" if torch.cuda.is_available() else "adamw_torch",
        warmup_steps=500,
        weight_decay=0.01,
        max_grad_norm=1.0,
        logging_first_step=True,
        load_best_model_at_end=False,  # Skip to avoid slowdowns
        # Memory optimizations for 8GB GPU
        tf32=True if torch.cuda.is_available() else False,  # Faster matmuls on Ampere+
        ddp_find_unused_parameters=False,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
    )
    trainer.train()
    trainer.save_model(cfg["log_dir"])

if __name__ == "__main__":
    main()
