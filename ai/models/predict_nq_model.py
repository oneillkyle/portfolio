
import tensorflow as tf
import numpy as np
from transformers import AutoTokenizer

MODEL_PATH = "ai/saved_models/trained_nq_model.keras"
TOKENIZER_PATH = "ai/tokenizer_model"
MAX_LENGTH = 75

model = tf.keras.models.load_model(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)

def decode_sequence(question_text):
    q_ids = tokenizer.encode(question_text, add_special_tokens=True, truncation=True, max_length=MAX_LENGTH)
    q_ids = tf.convert_to_tensor([q_ids + [0] * (MAX_LENGTH - len(q_ids))])
    a_input = tf.convert_to_tensor([[tokenizer.cls_token_id] + [0] * (MAX_LENGTH - 1)])
    pred = model.predict([q_ids, a_input])
    pred_ids = np.argmax(pred[0], axis=-1)
    return tokenizer.decode(pred_ids, skip_special_tokens=True)

while True:
    question = input("\nEnter question (or 'exit'): ").strip()
    if question.lower() == "exit":
        break
    print("💬 Answer:", decode_sequence(question))
