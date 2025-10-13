from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments
from datasets import Dataset
import os
from .utils import load_config, parse_args

def main():
    args = parse_args()
    cfg = load_config(args.config)
    model_name = cfg["model_name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load test data (first N lines or a separate test file if available)
    test_path = os.path.join(cfg["processed_dir"], "test.txt")
    if os.path.exists(test_path):
        with open(test_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
    else:
        # Fallback: use a portion of the raw data
        raw_path = os.path.join(cfg["raw_dir"], os.path.basename(cfg["raw_data_path"]))
        with open(raw_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for i, line in enumerate(f) if line.strip() and i % 20 == 0]
    dataset = Dataset.from_dict({"text": lines})

    # Tokenize and corrupt spans
    def preprocess(example):
        ids = tokenizer.encode(example["text"], truncation=True, max_length=int(cfg["block_size"]))
        return {"input_ids": ids, "labels": ids}
    dataset = dataset.map(preprocess, remove_columns=["text"])

    model_dir = cfg.get("log_dir", "./logs")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)

    training_args = TrainingArguments(
        output_dir=model_dir,
        per_device_eval_batch_size=int(cfg["batch_size"]),
        report_to=["tensorboard"],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=dataset,
    )
    metrics = trainer.evaluate()
    print(metrics)

if __name__ == "__main__":
    main()
