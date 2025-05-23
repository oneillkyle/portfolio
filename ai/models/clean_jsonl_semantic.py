
import json
import re
import html
import nltk
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer, util
from collections import Counter

nltk.download('punkt')

# HTML cleaning utility
html_tag_pattern = re.compile(r"<[^>]+>")

def clean_html(text):
    text = html.unescape(text)
    return re.sub(html_tag_pattern, " ", text)

# Load semantic similarity model
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
                document_text = clean_html(data.get("document_text", ""))
                tokens = document_text.split()
                annotations = data.get("annotations", [{}])
                annotation = annotations[0]
                long_answer = annotation.get("long_answer", {})
                short_answers = annotation.get("short_answers", [])

                answer = ""
                answer_type = "none"

                if short_answers:
                    sa = short_answers[0]
                    start = sa.get("start_token")
                    end = sa.get("end_token")
                    if start is not None and end is not None:
                        short = " ".join(tokens[start:end])
                        if retain_dual:
                            answer = short
                            answer_type = "short"
                        else:
                            answer = short.strip()
                            answer_type = "short"
                elif long_answer and "start_token" in long_answer and "end_token" in long_answer:
                    start = long_answer["start_token"]
                    end = long_answer["end_token"]
                    long = " ".join(tokens[start:end])
                    sentence = extract_best_sentence_bert(long, question)
                    if retain_dual:
                        answer = sentence
                        answer_type = "long"
                    else:
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
                    if retain_dual and short_answers and long_answer:
                        stats["both_used"] += 1
                        cleaned["short_answer"] = short.strip()
                        cleaned["long_answer"] = sentence.strip()
                    fout.write(json.dumps(cleaned) + "\n")
                else:
                    stats["skipped"] += 1
            except Exception:
                stats["skipped"] += 1
                continue

    print("✅ Cleaning complete.")
    print(f"Total kept: {stats['kept']}")
    print(f"Total skipped: {stats['skipped']}")
    print(f"Short used: {stats['answer_type']['short']}")
    print(f"Long used: {stats['answer_type']['long']}")
    print(f"Both used (dual mode): {stats['both_used']}")
    print(f"Avg question length: {sum(stats['question_lengths']) / max(1, len(stats['question_lengths'])):.2f}")
    print(f"Avg answer length: {sum(stats['answer_lengths']) / max(1, len(stats['answer_lengths'])):.2f}")
