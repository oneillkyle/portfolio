import tensorflow as tf
from transformers import AutoTokenizer

TFRECORD_PATH = "ai/datasets/cleaned_data.train.tfrecord"
TOKENIZER_PATH = "ai/tokenizer_model"
NUM_EXAMPLES = 5

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)

# Define parsing function
def parse_fn(example_proto):
    features = {
        "question": tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
        "answer": tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
    }
    return tf.io.parse_single_example(example_proto, features)

# Load and inspect
ds = tf.data.TFRecordDataset(TFRECORD_PATH).map(parse_fn)

print(f"🔍 Previewing first {NUM_EXAMPLES} examples in {TFRECORD_PATH}\n")
for i, ex in enumerate(ds.take(NUM_EXAMPLES)):
    q_ids = ex["question"].numpy().tolist()
    a_ids = ex["answer"].numpy().tolist()
    print(f"Example {i + 1}")
    print("Q:", tokenizer.decode(q_ids, skip_special_tokens=True))
    print("A:", tokenizer.decode(a_ids, skip_special_tokens=True))
    print("-" * 60)
