
import json
import argparse
from transformers import T5Tokenizer, T5ForConditionalGeneration
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

def load_data(path):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)
            data.append((obj['question'], obj['answer']))
    return data

def evaluate(model_dir, data_path, max_len=64, num_beams=4):
    tokenizer = T5Tokenizer.from_pretrained(model_dir)
    model = T5ForConditionalGeneration.from_pretrained(model_dir)

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    smoothie = SmoothingFunction().method4

    data = load_data(data_path)
    bleu_scores = []
    rouge1_scores = []
    rougeL_scores = []

    for question, reference in data:
        input_text = f"question: {question}"
        inputs = tokenizer(input_text, return_tensors='pt', truncation=True, padding=True, max_length=max_len)
        outputs = model.generate(
            **inputs,
            max_length=max_len,
            num_beams=num_beams,
            early_stopping=True
        )
        pred = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # BLEU
        bleu = sentence_bleu([reference.split()], pred.split(), smoothing_function=smoothie)
        bleu_scores.append(bleu)

        # ROUGE
        scores = scorer.score(reference, pred)
        rouge1_scores.append(scores['rouge1'].fmeasure)
        rougeL_scores.append(scores['rougeL'].fmeasure)

    n = len(data)
    print(f"Evaluated {n} examples")
    print(f"Average BLEU: {sum(bleu_scores)/n:.4f}")
    print(f"Average ROUGE-1: {sum(rouge1_scores)/n:.4f}")
    print(f"Average ROUGE-L: {sum(rougeL_scores)/n:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate T5 QA model")
    parser.add_argument("--model_dir", type=str, default="/ai/saved_models/t5_trained_nq", help="Path to fine-tuned T5 model")
    parser.add_argument("--data", type=str, required=True, help="Path to JSONL file for evaluation")
    parser.add_argument("--max_len", type=int, default=64, help="Maximum sequence length")
    parser.add_argument("--beams", type=int, default=4, help="Number of beams for generation")
    args = parser.parse_args()

    evaluate(args.model_dir, args.data, max_len=args.max_len, num_beams=args.beams)
