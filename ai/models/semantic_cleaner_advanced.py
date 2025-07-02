import os
import json
import re
import html
import nltk
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer, util
from collections import Counter

# Download and set up punkt tokenizer
nltk.download("punkt")
html_tag_pattern = re.compile(r"<[^>]+>")

def clean_html(text):
    text = html.unescape(text)
    return re.sub(html_tag_pattern, " ", text)

model = SentenceTransformer("all-MiniLM-L6-v2")

BAD_PHRASES = [
    "the united states", "the us", "america", "united states constitution"
]

def contains_repetitive_phrases(text):
    text_lc = text.lower()
    return any(phrase in text_lc for phrase in BAD_PHRASES)

def extract_best_sentence_bert(long_text, question):
    sentences = sent_tokenize(long_text)
    if not sentences:
        return ""
    question_emb = model.encode(question, convert_to_tensor=True)
    sent_embs = model.encode(sentences, convert_to_tensor=True)
    scores = util.cos_sim(question_emb, sent_embs)[0]
    best_idx = scores.argmax().item()
    return sentences[best_idx].strip()

def clean_nq_semantic(input_path, output_path, limit=100000):
    seen = set()
    stats = Counter()
    with open(input_path, "r", encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
        for i, line in enumerate(fin):
            if limit and i >= limit:
                break
            try:
                data = json.loads(line)
                q = clean_html(data.get("question_text", ""))
                doc = data.get("document_text", "")
                tokens = doc.split()
                ann = data.get("annotations", [{}])[0]
                sa = ann.get("short_answers", [])
                la = ann.get("long_answer", {})
                answer = ""

                if sa:
                    start, end = sa[0]["start_token"], sa[0]["end_token"]
                    answer = clean_html(" ".join(tokens[start:end]))
                elif "start_token" in la:
                    start, end = la["start_token"], la["end_token"]
                    context = " ".join(tokens[start:end])
                    answer = extract_best_sentence_bert(context, q)

                if len(answer.split()) < 3 or contains_repetitive_phrases(answer):
                    stats["filtered"] += 1
                    continue

                key = f"{q.strip()}|||{answer.strip()}"
                if key in seen:
                    stats["duplicate"] += 1
                    continue
                seen.add(key)

                stats["kept"] += 1
                fout.write(json.dumps({"question": q.strip(), "answer": answer.strip()}) + "\n")

                if stats["kept"] % 500 == 0:
                    print(f"✅ Kept: {stats['kept']}")

            except Exception as e:
                stats["error"] += 1
                print(f"⚠️ Error line {i}: {e}")
    print("Done.")
    print(stats)