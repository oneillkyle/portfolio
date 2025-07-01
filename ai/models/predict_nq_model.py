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
    q_ids = q_ids + [0] * (MAX_LENGTH - len(q_ids))
    q_input = tf.convert_to_tensor([q_ids])

    decoded_ids = [tokenizer.cls_token_id]
    for _ in range(MAX_LENGTH - 1):
        a_input = tf.convert_to_tensor([decoded_ids + [0] * (MAX_LENGTH - len(decoded_ids))])
        preds = model.predict([q_input, a_input], verbose=0)
        next_id = int(np.argmax(preds[0, len(decoded_ids) - 1]))
        if next_id == tokenizer.pad_token_id:
            break
        decoded_ids.append(next_id)

    print("Predicted token IDs:", decoded_ids)
    return tokenizer.decode(decoded_ids, skip_special_tokens=True)

# Optional: Inspect TFRecord content
raw = next(iter(tf.data.TFRecordDataset("ai/datasets/cleaned_data.train.tfrecord")))
parsed = tf.io.parse_single_example(raw, {
    "question": tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
    "answer": tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
})
print("Q IDs:", parsed["question"].numpy())
print("A IDs:", parsed["answer"].numpy())
print("Decoded Answer:", tokenizer.decode(parsed["answer"].numpy().tolist(), skip_special_tokens=True))

while True:
    question = input("\nEnter question (or 'exit'): ").strip()
    if question.lower() == "exit":
        break
    print("\U0001F4AC Answer:", decode_sequence(question))
