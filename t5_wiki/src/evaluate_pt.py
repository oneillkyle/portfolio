from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments
from datasets import load_dataset, Dataset
import numpy as np
import os
import yaml
from .utils import load_config, parse_args

def main():
    args = parse_args()
    cfg = load_config(args.config)
    model_name = cfg["model_name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load validation data (same split as train_pt.py)
    raw_path = os.path.join(cfg["raw_dir"], os.path.basename(cfg["raw_data_path"]))
    with open(raw_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    dataset = Dataset.from_dict({"text": lines})
    split = dataset.train_test_split(test_size=0.05, seed=cfg.get("random_seed", 42))
    val_ds = split["test"]

    # Use the latest checkpoint if available
    model_dir = cfg.get("log_dir", "./logs")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)

    # Define compute_metrics for perplexity
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        loss = np.mean((logits - labels) ** 2)  # Placeholder; HF Trainer will use its own loss
        ppl = np.exp(loss)
        return {"eval_loss": loss, "eval_perplexity": ppl}

    training_args = TrainingArguments(
        output_dir=model_dir,
        per_device_eval_batch_size=int(cfg["batch_size"]),
        report_to=["tensorboard"],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )
    metrics = trainer.evaluate()
    print(metrics)

if __name__ == "__main__":
    main()
