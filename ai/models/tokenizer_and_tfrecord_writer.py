import tensorflow as tf
import json
import random
from transformers import AutoTokenizer

INPUT_JSONL = "ai/datasets/cleaned_nq.jsonl"
TRAIN_TFRECORD = "ai/datasets/cleaned_data.train.tfrecord"
VAL_TFRECORD = "ai/datasets/cleaned_data.val.tfrecord"
TOKENIZER_PATH = "ai/tokenizer_model"
MAX_LENGTH = 75
VAL_SPLIT = 0.1

random.seed(42)
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
CLS_ID = tokenizer.cls_token_id
SEP_ID = tokenizer.sep_token_id

examples = []
with open(INPUT_JSONL, "r", encoding="utf-8") as f:
    for line in f:
        try:
            item = json.loads(line)
            question = item.get("question", "").strip()
            answer = item.get("answer", "").strip()

            if not question or not answer:
                continue

            q_ids = tokenizer.encode(question, add_special_tokens=True, truncation=True, max_length=MAX_LENGTH)
            a_ids = tokenizer.encode(answer, add_special_tokens=False, truncation=True, max_length=MAX_LENGTH - 1)
            a_ids.append(SEP_ID)  # add [SEP] as stop token

            examples.append((q_ids, a_ids))

        except Exception as e:
            print(f"⚠️ Skipped due to error: {e}")

print(f"✅ Loaded {len(examples)} total examples")
random.shuffle(examples)
val_size = int(len(examples) * VAL_SPLIT)
val_examples = examples[:val_size]
train_examples = examples[val_size:]


def write_tfrecord(examples, path):
    with tf.io.TFRecordWriter(path) as writer:
        for i, (q_ids, a_ids) in enumerate(examples):
            q_feature = tf.train.Feature(int64_list=tf.train.Int64List(value=q_ids))
            a_feature = tf.train.Feature(int64_list=tf.train.Int64List(value=a_ids))
            example = tf.train.Example(features=tf.train.Features(feature={
                "question": q_feature,
                "answer": a_feature
            }))
            writer.write(example.SerializeToString())
            if (i + 1) % 500 == 0:
                print(f"📝 Written {i + 1} examples to {path}")

write_tfrecord(train_examples, TRAIN_TFRECORD)
write_tfrecord(val_examples, VAL_TFRECORD)
print(f"✅ TFRecord creation complete: {len(train_examples)} train, {len(val_examples)} validation")
