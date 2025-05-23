
import json
import re
import html
import nltk
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt')

html_tag_pattern = re.compile(r"<[^>]+>")

def clean_html(text):
    text = html.unescape(text)
    return re.sub(html_tag_pattern, " ", text)

def extract_best_sentence(long_text, question):
    sentences = sent_tokenize(long_text)
    if not sentences:
        return ""
    corpus = [question] + sentences
    vectorizer = TfidfVectorizer().fit_transform(corpus)
    similarity = cosine_similarity(vectorizer[0:1], vectorizer[1:]).flatten()
    best_index = similarity.argmax()
    return sentences[best_index].strip()

def clean_nq_with_sentence_extraction(input_path, output_path, limit=100000):
    with open(input_path, "r", encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
        count = 0
        for i, line in enumerate(fin):
            if limit and i >= limit:
                break
            try:
                data = json.loads(line)
                question = clean_html(data.get("question_text", ""))
                long_text = clean_html(data.get("document_text", ""))
                annotations = data.get("annotations", [{}])
                long_answer = annotations[0].get("long_answer", {})
                short_answers = annotations[0].get("short_answers", [])

                if short_answers:
                    sa = short_answers[0]
                    start = sa.get("start_token")
                    end = sa.get("end_token")
                    if start is not None and end is not None:
                        tokens = long_text.split()
                        short = " ".join(tokens[start:end])
                        answer = short.strip()
                else:
                    if long_answer and "start_token" in long_answer and "end_token" in long_answer:
                        start = long_answer["start_token"]
                        end = long_answer["end_token"]
                        tokens = long_text.split()
                        long = " ".join(tokens[start:end])
                        answer = extract_best_sentence(long, question)
                    else:
                        answer = ""

                if len(question) > 5 and len(answer.split()) > 3 and answer.lower() not in question.lower():
                    cleaned = {
                        "question": question.strip(),
                        "answer": answer.strip()
                    }
                    fout.write(json.dumps(cleaned) + "\n")
                    count += 1
            except Exception:
                continue
        print(f"✅ Saved {count} cleaned Q/A pairs to {output_path}")

if __name__ == "__main__":
    clean_nq_with_sentence_extraction("simplified-nq-train.jsonl", "cleaned_nq.jsonl")
