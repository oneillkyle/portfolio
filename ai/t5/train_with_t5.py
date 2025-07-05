import json
import random
from datasets import load_dataset, Dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments, DataCollatorForSeq2Seq
import torch

MODEL_NAME = "../saved_models/t5_wiki_pretrain_fast"
MAX_SOURCE_LEN = 64
MAX_TARGET_LEN = 64
TRAIN_PATH = "../datasets/cleaned_nq.train.jsonl"
VAL_PATH = "../datasets/cleaned_nq.val.jsonl"
MODEL_OUT = "../saved_models/t5_trained_nq"
DS_CONFIG = "../../deepspeed_config.json"


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def tokenize(example, tokenizer):
    source = f"question: {example['question']}"
    target = example["answer"]
    source_enc = tokenizer(source, max_length=MAX_SOURCE_LEN,
                           truncation=True, padding="max_length")
    target_enc = tokenizer(target, max_length=MAX_TARGET_LEN,
                           truncation=True, padding="max_length")
    return {
        "input_ids": source_enc["input_ids"],
        "attention_mask": source_enc["attention_mask"],
        "labels": target_enc["input_ids"]
    }


def main():
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)

    train_data = load_jsonl(TRAIN_PATH)
    val_data = load_jsonl(VAL_PATH)

    dataset = Dataset.from_list(train_data + val_data)
    dataset = dataset.train_test_split(test_size=0.1)
    train_ds = dataset["train"]
    val_ds = dataset["test"]

    train_ds = train_ds.map(lambda x: tokenize(x, tokenizer))
    val_ds = val_ds.map(lambda x: tokenize(x, tokenizer))

    args = TrainingArguments(
        output_dir=MODEL_OUT,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        eval_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=10,
        logging_dir=f"{MODEL_OUT}/logs",
        save_total_limit=2,
        load_best_model_at_end=True,
        fp16=torch.cuda.is_available(),
        deepspeed=DS_CONFIG,
    )

    # training_args = TrainingArguments(
    #     output_dir=MODEL_OUT,
    #     per_device_train_batch_size=16,
    #     per_device_eval_batch_size=16,
    #     num_train_epochs=10,
    #     do_train=True,
    #     do_eval=True,
    #     evaluate_during_training=True,  # legacy flag
    #     logging_dir=f"{MODEL_OUT}/logs",
    #     logging_steps=500,
    #     save_steps=500,
    #     save_total_limit=2,
    #     fp16=torch.cuda.is_available(),
    # )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model)
    )

    trainer.train()
    metrics = trainer.evaluate()
    print(f"** Final Eval Loss: {metrics['eval_loss']:.4f} **")
    trainer.save_model(MODEL_OUT)
    tokenizer.save_pretrained(MODEL_OUT)


if __name__ == "__main__":
    main()
