import os
import json
import re
import html
import nltk
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer, util
from collections import Counter

# Ensure nltk_data is available and punkt is downloaded
nltk_data_dir = os.path.expanduser("~/nltk_data")
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir)

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", download_dir=nltk_data_dir)

nltk.data.path.append(nltk_data_dir)

html_tag_pattern = re.compile(r"<[^>]+>")

def clean_html(text):
    text = html.unescape(text)
    return re.sub(html_tag_pattern, " ", text)

model = SentenceTransformer("all-MiniLM-L6-v2")

def extract_best_sentence_bert(long_text, question):
    sentences = sent_tokenize(long_text)
    if not sentences:
        return ""
    question_emb = model.encode(question, convert_to_tensor=True)
    sent_embs = model.encode(sentences, convert_to_tensor=True)
    scores = util.cos_sim(question_emb, sent_embs)[0]
    best_idx = scores.argmax().item()
    return sentences[best_idx].strip()

def clean_nq_semantic(input_path, output_path, limit=100000, retain_dual=False):
    stats = {
        "kept": 0,
        "skipped": 0,
        "short_used": 0,
        "long_used": 0,
        "both_used": 0,
        "answer_lengths": [],
        "question_lengths": [],
        "answer_type": Counter()
    }

    with open(input_path, "r", encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
        for i, line in enumerate(fin):
            if limit and i >= limit:
                break
            try:
                data = json.loads(line)
                question = clean_html(data.get("question_text", ""))
                raw_tokens = data.get("document_text", "").split()
                annotations = data.get("annotations", [{}])
                annotation = annotations[0]
                long_answer = annotation.get("long_answer", {})
                short_answers = annotation.get("short_answers", [])

                short, sentence = "", ""
                answer = ""
                answer_type = "none"

                if short_answers:
                    sa = short_answers[0]
                    start = sa.get("start_token")
                    end = sa.get("end_token")
                    if start is not None and end is not None and end > start:
                        short = " ".join(raw_tokens[start:end])
                        short = clean_html(short)
                        answer = short.strip()
                        answer_type = "short"

                if not short and long_answer and "start_token" in long_answer and "end_token" in long_answer:
                    start = long_answer["start_token"]
                    end = long_answer["end_token"]
                    if end > start:
                        long = " ".join(raw_tokens[start:end])
                        long = clean_html(long)
                        sentence = extract_best_sentence_bert(long, question)
                        answer = sentence.strip()
                        answer_type = "long"

                if len(question) > 5 and len(answer.split()) > 3 and answer.lower() not in question.lower():
                    stats["kept"] += 1
                    stats["answer_type"][answer_type] += 1
                    stats["question_lengths"].append(len(question.split()))
                    stats["answer_lengths"].append(len(answer.split()))
                    cleaned = {
                        "question": question.strip(),
                        "answer": answer.strip()
                    }
                    if retain_dual and short and sentence:
                        stats["both_used"] += 1
                        cleaned["short_answer"] = short.strip()
                        cleaned["long_answer"] = sentence.strip()
                    fout.write(json.dumps(cleaned) + "\n")
                    if stats["kept"] % 100 == 0:
                        print(f"📝 Written {stats['kept']} examples...")
                else:
                    stats["skipped"] += 1
            except Exception as e:
                stats["skipped"] += 1
                print(f"⚠️ Error at line {i}: {e}")
                continue

    fout.flush()
    print("✅ Cleaning complete.")
    print(f"Total kept: {stats['kept']}")
    print(f"Total skipped: {stats['skipped']}")
    print(f"Short used: {stats['answer_type']['short']}")
    print(f"Long used: {stats['answer_type']['long']}")
    print(f"Both used (dual mode): {stats['both_used']}")
    print(f"Avg question length: {sum(stats['question_lengths']) / max(1, len(stats['question_lengths'])):.2f}")
    print(f"Avg answer length: {sum(stats['answer_lengths']) / max(1, len(stats['answer_lengths'])):.2f}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Clean NQ dataset using semantic matching.")
    parser.add_argument("--input", type=str, default="ai/datasets/simplified-nq-train.jsonl", help="Path to input JSONL file")
    parser.add_argument("--output", type=str, default="ai/datasets/cleaned_nq.jsonl", help="Path to output cleaned JSONL")
    parser.add_argument("--limit", type=int, default=100000, help="Max examples to process")
    parser.add_argument("--dual", action="store_true", help="Include both short and long answers in output")

    args = parser.parse_args()

    clean_nq_semantic(
        input_path=args.input,
        output_path=args.output,
        limit=args.limit,
        retain_dual=args.dual
    )
