import json
import csv
import argparse
from pathlib import Path
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

def load_model(model_dir, device):
    tokenizer = T5Tokenizer.from_pretrained(model_dir)
    model = T5ForConditionalGeneration.from_pretrained(model_dir)
    model.to(device)
    model.eval()
    return tokenizer, model

def predict(question, tokenizer, model, device, max_length=64, num_beams=5):
    input_text = f"question: {question}"
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_length
    ).to(device)
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        num_beams=num_beams,
        early_stopping=True,
        no_repeat_ngram_size=2,
        num_return_sequences=1,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def process_jsonl(path, tokenizer, model, device, max_length, num_beams):
    results = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            q = obj.get("question", "")
            if not q:
                continue
            a = predict(q, tokenizer, model, device, max_length, num_beams)
            results.append({"question": q, "prediction": a})
    return results

def process_csv(path, tokenizer, model, device, max_length, num_beams):
    results = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            q = row.get("question", "")
            if not q:
                continue
            a = predict(q, tokenizer, model, device, max_length, num_beams)
            results.append({"question": q, "prediction": a})
    return results

def save_csv(results, out_path):
    if not results:
        print("No predictions to save.")
        return
    with open(out_path, "w", newline='', encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["question", "prediction"])
        writer.writeheader()
        writer.writerows(results)
    print(f"✅ Saved {len(results)} predictions to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch predict with fine-tuned T5 QA model")
    parser.add_argument("--model_dir", type=str, default="ai/saved_models/t5_trained_nq", help="Path to fine-tuned T5 model")
    parser.add_argument("--input", type=str, required=True, help="Input JSONL or CSV file with 'question' field")
    parser.add_argument("--output", type=str, default="ai/datasets/predictions.csv", help="Output CSV file")
    parser.add_argument("--max_length", type=int, default=64, help="Max sequence length for generation")
    parser.add_argument("--num_beams", type=int, default=5, help="Number of beams for beam search")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
    args = parser.parse_args()

    tokenizer, model = load_model(args.model_dir, args.device)
    print(f"Loaded model from {args.model_dir} on {args.device}")

    input_path = Path(args.input)
    if input_path.suffix == ".jsonl":
        results = process_jsonl(args.input, tokenizer, model, args.device, args.max_length, args.num_beams)
    elif input_path.suffix == ".csv":
        results = process_csv(args.input, tokenizer, model, args.device, args.max_length, args.num_beams)
    else:
        raise ValueError("Input must be .jsonl or .csv")

    save_csv(results, args.output)