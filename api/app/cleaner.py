from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse
import os, json, tempfile
from pydantic import BaseModel
from typing import Optional
import nltk
import html
import re
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer, util

app = FastAPI()

# Ensure punkt tokenizer is available
nltk_data_dir = os.path.expanduser("~/nltk_data")
os.makedirs(nltk_data_dir, exist_ok=True)
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

class CleaningRequest(BaseModel):
    limit: Optional[int] = 1000
    dual: Optional[bool] = False

@app.post("/clean")
async def clean_nq_file(file: UploadFile, config: CleaningRequest):
    try:
        contents = await file.read()
        input_lines = contents.decode("utf-8").splitlines()
        
        results = []
        for i, line in enumerate(input_lines):
            if config.limit and i >= config.limit:
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

                short, sentence = "", ""
                answer = ""

                if short_answers:
                    sa = short_answers[0]
                    start = sa.get("start_token")
                    end = sa.get("end_token")
                    if start is not None and end is not None:
                        short = " ".join(tokens[start:end])
                        answer = short.strip()

                if not short and long_answer and "start_token" in long_answer and "end_token" in long_answer:
                    start = long_answer["start_token"]
                    end = long_answer["end_token"]
                    long = " ".join(tokens[start:end])
                    sentence = extract_best_sentence_bert(long, question)
                    answer = sentence.strip()

                if len(question) > 5 and len(answer.split()) > 3 and answer.lower() not in question.lower():
                    item = {
                        "question": question.strip(),
                        "answer": answer.strip()
                    }
                    if config.dual and short and sentence:
                        item["short_answer"] = short.strip()
                        item["long_answer"] = sentence.strip()
                    results.append(item)

            except Exception as e:
                continue

        return JSONResponse(content={"cleaned": results[:config.limit], "total": len(results)})

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
