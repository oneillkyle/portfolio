import argparse
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from sentence_transformers import SentenceTransformer, util

from ai.elastic.passages import retrieve

passages = []
sbert = SentenceTransformer("all-MiniLM-L6-v2")
passage_embeddings = sbert.encode(passages, convert_to_tensor=True)


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
        max_length=64,
        num_beams=8,
        early_stopping=True,
        no_repeat_ngram_size=3,
        length_penalty=1.2,
        repetition_penalty=2.0,
        top_p=0.9,              # nucleus sampling
        do_sample=True,         # add some diversity
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def predict_with_es(question, tokenizer, model, device, max_length=64, num_beams=8, k=3):
    ctxs = retrieve(question, k)
    context = " ".join(ctxs)
    prompt = f"context: {context}  question: {question}"

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_length
    ).to(device)

    outputs = model.generate(
        **inputs,
        max_length=64,
        num_beams=num_beams,
        early_stopping=True,
        no_repeat_ngram_size=3,
        length_penalty=1.2,
        repetition_penalty=2.0,
        top_p=0.9,              # nucleus sampling
        do_sample=True,         # add some diversity
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(
        description="Interactive T5 QA Prediction")
    parser.add_argument(
        "--model_dir", type=str, default="ai/saved_models/t5_trained_nq",
        help="Directory of the fine-tuned T5 model"
    )
    parser.add_argument(
        "--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run the model on"
    )
    parser.add_argument(
        "--max_length", type=int, default=64,
        help="Maximum sequence length for generation"
    )
    parser.add_argument(
        "--num_beams", type=int, default=5,
        help="Number of beams for beam search"
    )
    args = parser.parse_args()

    tokenizer, model = load_model(args.model_dir, args.device)
    print(f"Loaded T5 model from '{args.model_dir}' on {args.device}")

    while True:
        question = input("\nEnter question (or 'exit'): ").strip()
        if question.lower() in ("exit", "quit"):
            break
        answer = predict_with_es(question, tokenizer, model,
                         args.device, args.max_length, args.num_beams)
        print("Answer:", answer)


if __name__ == "__main__":
    main()
