
import json
import csv
import argparse
from pathlib import Path
from transformers import AutoTokenizer
import tensorflow as tf
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

MODEL_PATH = "trained_nq_model"
TOKENIZER_PATH = "tokenizer_model"
MAX_LENGTH = 75

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
model = tf.keras.models.load_model(MODEL_PATH)
rouge = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
smoothie = SmoothingFunction().method4

def predict(question):
    q_ids = tokenizer.encode(question, add_special_tokens=True, truncation=True, max_length=MAX_LENGTH)
    q_input = tf.convert_to_tensor([q_ids + [0] * (MAX_LENGTH - len(q_ids))])
    a_input = tf.convert_to_tensor([[tokenizer.cls_token_id] + [0] * (MAX_LENGTH - 1)])
    pred = model.predict([q_input, a_input])
    pred_ids = np.argmax(pred[0], axis=-1)
    return tokenizer.decode(pred_ids, skip_special_tokens=True)

def evaluate(gold, pred):
    rouge_scores = rouge.score(gold, pred)
    bleu = sentence_bleu([gold.split()], pred.split(), smoothing_function=smoothie)
    return rouge_scores, bleu

def process_jsonl(path):
    results = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            question = data["question"]
            gold = data.get("answer", "")
            pred = predict(question)
            rouge_scores, bleu = evaluate(gold, pred)
            results.append({
                "question": question,
                "answer": gold,
                "predicted": pred,
                "bleu": bleu,
                "rouge1": rouge_scores["rouge1"].fmeasure,
                "rougeL": rouge_scores["rougeL"].fmeasure
            })
    return results

def process_csv(path):
    results = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            question = row["question"]
            gold = row.get("answer", "")
            pred = predict(question)
            rouge_scores, bleu = evaluate(gold, pred)
            results.append({
                "question": question,
                "answer": gold,
                "predicted": pred,
                "bleu": bleu,
                "rouge1": rouge_scores["rouge1"].fmeasure,
                "rougeL": rouge_scores["rougeL"].fmeasure
            })
    return results

def save_csv(results, out_path):
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Input JSONL or CSV file")
    parser.add_argument("--output", type=str, default="predictions.csv", help="CSV file to save results")
    args = parser.parse_args()

    input_path = Path(args.input)
    if input_path.suffix == ".jsonl":
        output = process_jsonl(input_path)
    elif input_path.suffix == ".csv":
        output = process_csv(input_path)
    else:
        raise ValueError("Input must be a .jsonl or .csv file")

    save_csv(output, args.output)
    print(f"✅ Saved predictions to {args.output}")
