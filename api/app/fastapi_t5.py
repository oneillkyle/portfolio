from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

app = FastAPI()

class QARequest(BaseModel):
    question: str

# Load model on startup
MODEL_DIR = "app/saved_models/t5_trained_nq"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = T5Tokenizer.from_pretrained(MODEL_DIR)
model = T5ForConditionalGeneration.from_pretrained(MODEL_DIR).to(DEVICE)
model.eval()

def predict(question: str, max_length: int = 64, num_beams: int = 5):
    input_text = f"question: {question}"
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_length
    ).to(DEVICE)
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        num_beams=num_beams,
        early_stopping=True,
        no_repeat_ngram_size=2,
        num_return_sequences=1,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

@app.post("/predict")
async def predict_endpoint(req: QARequest):
    q = req.question.strip()
    if not q:
        raise HTTPException(status_code=400, detail="Empty question")
    answer = predict(q)
    return {"question": q, "answer": answer}

@app.post("/batch_predict")
async def batch_predict(requests: list[QARequest]):
    results = []
    for req in requests:
        q = req.question.strip()
        if not q:
            results.append({"question": q, "answer": ""})
        else:
            results.append({"question": q, "answer": predict(q)})
    return results