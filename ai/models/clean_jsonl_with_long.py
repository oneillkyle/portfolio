
import json
import re
import html

# HTML cleaning utility
html_tag_pattern = re.compile(r"<[^>]+>")

def clean_html(text):
    text = html.unescape(text)
    return re.sub(html_tag_pattern, " ", text)

# Clean and filter dataset, include long answers
def clean_jsonl(input_path, output_path, limit=100000):
    with open(input_path, "r", encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
        for i, line in enumerate(fin):
            if limit and i >= limit:
                break
            try:
                data = json.loads(line)
                question = clean_html(data.get("question_text", ""))
                document_text = clean_html(data.get("document_text", ""))
                annotations = data.get("annotations", [{}])
                
                short_answers = annotations[0].get("short_answers", [])
                long_answer = annotations[0].get("long_answer", {})
                
                short_answer = ""
                if short_answers:
                    sa = short_answers[0]
                    start = sa.get("start_token")
                    end = sa.get("end_token")
                    if start is not None and end is not None:
                        short_answer = " ".join(document_text.split()[start:end])

                long_answer_text = ""
                if long_answer and "start_token" in long_answer and "end_token" in long_answer:
                    l_start = long_answer["start_token"]
                    l_end = long_answer["end_token"]
                    long_answer_text = " ".join(document_text.split()[l_start:l_end])

                if len(question) > 3 and (len(short_answer) > 1 or len(long_answer_text) > 1):
                    cleaned = {
                        "question": question,
                        "short_answer": short_answer,
                        "long_answer": long_answer_text
                    }
                    fout.write(json.dumps(cleaned) + "\n")
            except Exception:
                continue

if __name__ == "__main__":
    input_path = "ai/datasets/simplified-nq-train.jsonl"
    output_path = "ai/datasets/cleaned_nq.jsonl"
    clean_jsonl(input_path, output_path)
