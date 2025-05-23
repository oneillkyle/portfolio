import json
import re
import html
import tensorflow as tf
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, normalizers
from tokenizers.normalizers import NFD, Lowercase, StripAccents, Sequence
from tokenizers.pre_tokenizers import Whitespace

# HTML cleaning utility
html_tag_pattern = re.compile(r"<[^>]+>")


def clean_html(text):
    text = html.unescape(text)
    return re.sub(html_tag_pattern, " ", text)


# Define filters to eliminate bad samples
generic_answers = {
    "yes", "no", "none", "n/a", "not available", "unknown", "inches", "centimeters", "meters"
}

bad_starts = ("is the", "was the", "has been", "it is",
              "there are", "they are", "this is")


def is_valid_pair(q, a):
    q = q.strip()
    a = a.strip()

    if len(q) <= 3 or len(a) <= 5:
        return False
    if q.lower() == a.lower():
        return False
    if a.lower() in q.lower():
        return False
    if len(a.split()) < 3:
        return False
    if a.lower() in generic_answers:
        return False
    if a.lower().startswith(bad_starts):
        return False
    return True


def load_cleaned_texts(jsonl_path, limit=100000):
    texts = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            try:
                data = json.loads(line)
                q = clean_html(data.get("question", ""))
                a = clean_html(data.get("short_answer", "")
                               or data.get("long_answer", ""))
                if is_valid_pair(q, a):
                    texts.extend([q, a])
            except Exception:
                continue
    return texts


def train_and_save_tokenizer(texts, vocab_size=10000, path="tokenizer.json"):
    tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))
    tokenizer.normalizer = Sequence([NFD(), Lowercase(), StripAccents()])
    tokenizer.pre_tokenizer = Whitespace()
    trainer = trainers.WordPieceTrainer(
        vocab_size=vocab_size,
        special_tokens=["[PAD]", "[UNK]", "[START]", "[END]"]
    )
    tokenizer.train_from_iterator(texts, trainer)
    tokenizer.save(path)
    return tokenizer


def encode_example(q, a, tokenizer):
    q_ids = tokenizer.encode(q).ids
    a_ids = tokenizer.encode(a).ids
    start = tokenizer.token_to_id("[START]")
    end = tokenizer.token_to_id("[END]")
    q_ids = [start] + q_ids + [end]
    a_ids = [start] + a_ids + [end]

    def _int64_feature(val):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=val))

    features = {
        "question": _int64_feature(q_ids),
        "answer": _int64_feature(a_ids),
    }
    return tf.train.Example(features=tf.train.Features(feature=features))


def write_tfrecord_from_jsonl(jsonl_path, tokenizer, tfrecord_path="cleaned_data.tfrecord", limit=100000):
    count = 0
    with tf.io.TFRecordWriter(tfrecord_path) as writer, open(jsonl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            try:
                data = json.loads(line)
                q = clean_html(data.get("question", ""))
                a = clean_html(data.get("short_answer", "")
                               or data.get("long_answer", ""))
                if "is the current" in a.lower():
                    print("⚠️ Suspicious answer:", a)
                if is_valid_pair(q, a):
                    print(f"[{count}]")
                    print("Q raw:", q)
                    print("A raw:", a)
                    print("Q tokens:", tokenizer.encode(q).tokens)
                    print("A tokens:", tokenizer.encode(a).tokens)
                    example = encode_example(q, a, tokenizer)
                    writer.write(example.SerializeToString())
                    count += 1
            except Exception:
                continue
    print(f"✅ Wrote {count} valid examples to {tfrecord_path}")


if __name__ == "__main__":
    jsonl_input_path = "ai/datasets/cleaned_nq.jsonl"
    tfrecord_output_path = "ai/datasets/cleaned_data.tfrecord"
    tokenizer_output_path = "ai/tokenizers/nq_tokenizer.json"

    texts = load_cleaned_texts(jsonl_input_path)
    tokenizer = train_and_save_tokenizer(texts, path=tokenizer_output_path)
    write_tfrecord_from_jsonl(
        jsonl_input_path, tokenizer, tfrecord_path=tfrecord_output_path)
