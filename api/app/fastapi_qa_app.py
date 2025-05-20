
from fastapi import FastAPI, Request
from pydantic import BaseModel
from tokenizers import Tokenizer
import tensorflow as tf

app = FastAPI()

# Load model and tokenizer
model = tf.keras.models.load_model("api/app/saved_models/nq_model")
tokenizer = Tokenizer.from_file("api/app/tokenizers/nq_tokenizer.json")

sequence_length = 50
start_token = tokenizer.token_to_id("[START]")
end_token = tokenizer.token_to_id("[END]")

class QuestionRequest(BaseModel):
    question: str

def encode_input(text):
    ids = tokenizer.encode(text).ids
    ids = [start_token] + ids + [end_token]
    ids = ids[:sequence_length]
    ids += [0] * (sequence_length - len(ids))
    return tf.convert_to_tensor([ids], dtype=tf.int32)

def predict_answer(question_text):
    encoder_input = encode_input(question_text)
    decoder_input = tf.convert_to_tensor([[start_token] + [0] * (sequence_length - 1)], dtype=tf.int32)

    for i in range(1, sequence_length):
        predictions = model([encoder_input, decoder_input], training=False)
        predicted_id = tf.argmax(predictions[0, i - 1]).numpy()
        if predicted_id == end_token:
            break
        decoder_input = tf.tensor_scatter_nd_update(
            decoder_input,
            indices=[[0, i]],
            updates=[predicted_id]
        )

    token_ids = decoder_input[0].numpy()[1:i]
    return tokenizer.decode(token_ids)

@app.post("/predict")
async def predict(req: QuestionRequest):
    question = req.question.strip()
    if not question:
        return {"error": "Empty question"}
    answer = predict_answer(question)
    return {"question": question, "answer": answer}
