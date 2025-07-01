from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import AutoTokenizer
import tensorflow as tf
import numpy as np

MODEL_PATH = "api/app/saved_models/trained_nq_model.keras"
TOKENIZER_PATH = "api/app/tokenizer_model"
MAX_LENGTH = 75

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, local_files_only=True)
model = tf.keras.models.load_model(MODEL_PATH)

app = FastAPI()

class QARequest(BaseModel):
    question: str

@app.post("/predict")
async def predict_answer(payload: QARequest):
    try:
        q_ids = tokenizer.encode(payload.question, add_special_tokens=True, truncation=True, max_length=MAX_LENGTH)
        q_input = tf.convert_to_tensor([q_ids + [0] * (MAX_LENGTH - len(q_ids))])
        a_input = tf.convert_to_tensor([[tokenizer.cls_token_id] + [0] * (MAX_LENGTH - 1)])
        pred = model.predict([q_input, a_input])
        pred_ids = np.argmax(pred[0], axis=-1)
        decoded = tokenizer.decode(pred_ids)
        print("Predicted token IDs:", pred_ids)
        return {"answer": decoded.strip()}
    except Exception as e:
        return {"error": str(e)}

@app.get("/")
async def root():
    return {"message": "NQ Answering API Ready"}
