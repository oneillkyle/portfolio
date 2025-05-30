
import json
import random
from pathlib import Path
from transformers import AutoTokenizer
import tensorflow as tf
from tqdm import tqdm

INPUT_JSONL = "ai/datasets/cleaned_nq.jsonl"
TRAIN_TFRECORD = "ai/datasets/cleaned_data.train.tfrecord"
VAL_TFRECORD = "ai/datasets/cleaned_data.val.tfrecord"
TOKENIZER_PATH = "ai/tokenizer_model"
VALIDATION_SPLIT = 0.1
MAX_LENGTH = 75
MODEL_NAME = "bert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def serialize_example(q_ids, a_ids):
    feature = {
        "question": tf.train.Feature(int64_list=tf.train.Int64List(value=q_ids)),
        "answer": tf.train.Feature(int64_list=tf.train.Int64List(value=a_ids)),
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    return example.SerializeToString()


def tokenize_pair(question, answer):
    q_enc = tokenizer.encode(
        question, add_special_tokens=True, truncation=True, max_length=MAX_LENGTH)
    a_enc = tokenizer.encode(
        answer, add_special_tokens=True, truncation=True, max_length=MAX_LENGTH)
    return q_enc, a_enc


def write_tfrecord(data, out_path):
    count = 0
    with tf.io.TFRecordWriter(out_path) as writer:
        for item in tqdm(data, desc=f"Writing {out_path}"):
            try:
                q, a = item.get("question"), item.get("answer")
                if not q or not a or len(a.strip()) < 2:
                    continue
                q_ids, a_ids = tokenize_pair(q, a)
                if len(q_ids) == 0 or len(a_ids) == 0:
                    continue
                writer.write(serialize_example(q_ids, a_ids))
                count += 1
            except Exception as e:
                print(f"Skipping item due to error: {e}")
    print(f"✅ Wrote {count} examples to {out_path}")


def main():
    with open(INPUT_JSONL, "r", encoding="utf-8") as f:
        lines = [json.loads(l.strip()) for l in f if l.strip()]

    random.shuffle(lines)
    split_idx = int(len(lines) * (1 - VALIDATION_SPLIT))
    train_data = lines[:split_idx]
    val_data = lines[split_idx:]

    print(
        f"Total: {len(lines)} | Train: {len(train_data)} | Val: {len(val_data)}")
    write_tfrecord(train_data, TRAIN_TFRECORD)
    write_tfrecord(val_data, VAL_TFRECORD)

    # Save tokenizer config
    tokenizer.save_pretrained(TOKENIZER_PATH)
    print("✅ Tokenizer saved to tokenizer_model/")


if __name__ == "__main__":
    main()
