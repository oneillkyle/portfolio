#!/usr/bin/env python3
import math
from datasets import load_dataset
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
import torch

# ⚙️ Config
SUBSAMPLED_CORPUS = "../datasets/wiki_corpus.subsample.txt"
MODEL_NAME = "t5-small"
OUTPUT_DIR = "../saved_models/t5_wiki_pretrain_fast"
MAX_LENGTH = 256      # shorter sequence
BLOCK_SIZE = 256
BATCH_SIZE = 4        # per device
EPOCHS = 1
DS_CONFIG = "../../deepspeed_config.json"

# 1) Load & tokenize
ds = load_dataset("text", data_files={
                  "train": SUBSAMPLED_CORPUS}, split="train")
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)


def tokenize_fn(ex):
    return tokenizer(
        ex["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH
    )


tokenized = ds.map(tokenize_fn, batched=False, remove_columns=["text"])

# 2) Group into blocks of BLOCK_SIZE


def group_fn(examples):
    all_ids = sum(examples["input_ids"], [])
    total_len = (len(all_ids) // BLOCK_SIZE) * BLOCK_SIZE
    chunks = [all_ids[i: i + BLOCK_SIZE]
              for i in range(0, total_len, BLOCK_SIZE)]
    return {"input_ids": chunks}


lm_dataset = tokenized.map(group_fn, batched=True, batch_size=1000)

# 3) Model + checkpointing
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
model.gradient_checkpointing_enable()

# 4) Data collator (span-masking)
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # T5 uses span-masking internally with <extra_id_*> tokens
)

# 5) TrainingArguments with DeepSpeed & FP16
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=math.ceil(32 / BATCH_SIZE),  # simulate bs=32
    num_train_epochs=EPOCHS,
    fp16=True,
    logging_steps=1000,
    save_steps=5000,
    save_total_limit=2,
    deepspeed=DS_CONFIG,      # enable ZeRO stage 2 + fp16
    optim="adamw_torch",
)

# 6) Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

# 7) Train!
if __name__ == "__main__":
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
