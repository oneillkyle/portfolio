from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import Dataset
import numpy as np
import os
from t5_wiki.src.utils import load_config, parse_args
from tqdm import tqdm

# Optional: install rouge_score for ROUGE metric

try:
    from rouge_score import rouge_scorer
except ImportError:
    rouge_scorer = None
has_rouge = rouge_scorer is not None

def compute_rouge(preds, refs):
    if not has_rouge:
        return {}
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    scores = [scorer.score(r, p) for r, p in zip(refs, preds)]
    avg = {k: np.mean([s[k].fmeasure for s in scores]) for k in scores[0]}
    return avg

def main():
    args = parse_args()
    cfg = load_config(args.config)
    model_name = cfg["model_name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model_dir = cfg.get("log_dir", "./logs")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
    model.eval()
    
    # Set up logging to file
    log_file = os.path.join(model_dir, "advanced_eval_results.txt")
    os.makedirs(model_dir, exist_ok=True)
    
    def log_and_print(message):
        print(message)
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(message + "\n")
    
    log_and_print(f"Advanced Evaluation Results - Model: {model_name}")
    log_and_print("=" * 50)

    # Load test data
    test_path = os.path.join(cfg["processed_dir"], "test.txt")
    if os.path.exists(test_path):
        with open(test_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
    else:
        raw_path = os.path.join(cfg["raw_dir"], os.path.basename(cfg["raw_data_path"]))
        with open(raw_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for i, line in enumerate(f) if line.strip() and i % 20 == 0]
    dataset = Dataset.from_dict({"text": lines})

    # Generate outputs
    preds, refs = [], []
    df = dataset.to_pandas()
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Generating outputs"):
        input_ids = tokenizer.encode(row["text"], return_tensors="pt", truncation=True, max_length=int(cfg["block_size"]))
        output = model.generate(input_ids, max_length=int(cfg["block_size"]))
        pred = tokenizer.decode(output[0], skip_special_tokens=True)
        preds.append(pred)
        refs.append(row["text"])
        # Print and log a few samples
        if len(preds) <= 5:
            sample_msg = f"\nSample {len(preds)}:\nInput: {row['text']}\nOutput: {pred}\n"
            log_and_print(sample_msg)

    # Compute ROUGE if available
    if has_rouge:
        rouge = compute_rouge(preds, refs)
        rouge_msg = f"ROUGE Scores: {rouge}"
        log_and_print(rouge_msg)
    else:
        log_and_print("Install rouge_score for ROUGE metrics: pip install rouge_score")
    
    log_and_print(f"\nEvaluation completed. Total samples: {len(preds)}")
    log_and_print(f"Results saved to: {log_file}")

if __name__ == "__main__":
    main()
