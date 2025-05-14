from tensorflow.keras import layers, Model
import tensorflow as tf
import json
import re
import html
from tensorflow.keras.layers import TextVectorization

# === HTML cleaning utilities ===
html_tag_pattern = re.compile(r"<[^>]+>")


def clean_html(text):
    text = html.unescape(text)  # Converts &#x27; etc. to '
    return re.sub(html_tag_pattern, " ", text)

# === TensorFlow-compatible parser ===


def parse_json_line(line):
    try:
        data = json.loads(line.numpy().decode("utf-8"))
        question = clean_html(data.get("question_text", ""))
        document_text = clean_html(data.get("document_text", ""))
        short_answers = data.get("annotations", [{}])[
            0].get("short_answers", [])
        answer = ""
        if short_answers:
            sa = short_answers[0]
            start = sa.get("start_token")
            end = sa.get("end_token")
            if start is not None and end is not None:
                answer = " ".join(document_text.split()[start:end])
        return question or "", answer or ""
    except Exception:
        return "", ""


def tf_parse_json_line(line):
    question, answer = tf.py_function(
        parse_json_line, [line], [tf.string, tf.string])
    question.set_shape([])
    answer.set_shape([])
    return question, answer

# === Load and clean the dataset ===


def load_jsonl_dataset(file_path):
    dataset = tf.data.TextLineDataset(file_path)
    return dataset.map(tf_parse_json_line)

# === Filter logic ===


def is_valid_pair(q, a):
    return tf.logical_and(
        tf.strings.length(q) > 3,
        tf.strings.length(a) > 1
    )

# === Debugging utility ===


def validate_tensor(tensor, *, expected_rank=None, expected_shape=None, expected_dtype=None, label=""):
    if expected_rank is not None:
        tf.debugging.assert_rank(
            tensor, expected_rank, message=f"{label} expected rank {expected_rank}, got {tensor.shape}")
    if expected_shape is not None:
        tf.debugging.assert_shapes(
            [(tensor, expected_shape)], message=f"{label} expected shape {expected_shape}, got {tensor.shape}")
    if expected_dtype is not None:
        tf.debugging.assert_type(
            tensor, expected_dtype, message=f"{label} expected dtype {expected_dtype}, got {tensor.dtype}")


def debug_dataset_shapes(dataset, name="dataset", num_samples=1,
                         expected_input_shape=None, expected_input_dtype=None,
                         expected_label_shape=None, expected_label_dtype=None):
    print(f"\n🧪 Inspecting '{name}'...")
    for i, sample in enumerate(dataset.take(num_samples)):
        print(f"\n🔹 Sample {i + 1}")
        if isinstance(sample, tuple):
            inputs, labels = sample
            if isinstance(inputs, tuple):
                for j, inp in enumerate(inputs):
                    print(
                        f"  🔸 Input {j + 1}: shape = {inp.shape}, dtype = {inp.dtype}")
                    validate_tensor(inp, expected_rank=len(expected_input_shape), expected_shape=expected_input_shape,
                                    expected_dtype=expected_input_dtype, label=f"Input {j + 1}")
            else:
                print(
                    f"  🔸 Input: shape = {inputs.shape}, dtype = {inputs.dtype}")
                validate_tensor(inputs, expected_rank=len(
                    expected_input_shape), expected_shape=expected_input_shape, expected_dtype=expected_input_dtype, label="Input")
            print(f"  🔸 Label: shape = {labels.shape}, dtype = {labels.dtype}")
            validate_tensor(labels, expected_rank=len(expected_label_shape),
                            expected_shape=expected_label_shape, expected_dtype=expected_label_dtype, label="Label")
        else:
            print(
                f"  🔸 Output: shape = {sample.shape}, dtype = {sample.dtype}")
            validate_tensor(sample, expected_rank=len(expected_label_shape),
                            expected_shape=expected_label_shape, expected_dtype=expected_label_dtype, label="Output")


# === Parameters ===
file_path = 'ai/datasets/simplified-nq-train.jsonl'
max_tokens = 10000
sequence_length = 50
START_TOKEN = 1

# === Load & clean ===
raw_dataset = load_jsonl_dataset(file_path)
filtered_dataset = raw_dataset.filter(is_valid_pair)

# === Vectorizers ===
question_vectorizer = TextVectorization(
    max_tokens=max_tokens, output_sequence_length=sequence_length)
answer_vectorizer = TextVectorization(
    max_tokens=max_tokens, output_sequence_length=sequence_length)

question_ds = filtered_dataset.map(lambda q, a: q).batch(256)
answer_ds = filtered_dataset.map(lambda q, a: a).batch(256)

print("🧠 Adapting question vectorizer...")
question_vectorizer.adapt(question_ds)

print("🧠 Adapting answer vectorizer...")
answer_vectorizer.adapt(answer_ds)

# === Vectorize dataset ===
vectorized_dataset = filtered_dataset.map(
    lambda q, a: (question_vectorizer(q), answer_vectorizer(a)))

# === Decoder prep ===


def add_decoder_inputs_and_targets(question, answer):
    start_tokens = tf.fill([tf.shape(answer)[0], 1], START_TOKEN)
    decoder_input = tf.concat([start_tokens, answer[:, :-1]], axis=1)
    return (question, decoder_input), answer


batched_dataset = vectorized_dataset.batch(
    32).map(add_decoder_inputs_and_targets)
batched_dataset = batched_dataset.prefetch(tf.data.AUTOTUNE)

# === Debug pipeline ===
debug_dataset_shapes(
    batched_dataset,
    name="Training dataset",
    expected_input_shape=[None, sequence_length],
    expected_input_dtype=tf.int32,
    expected_label_shape=[None, sequence_length],
    expected_label_dtype=tf.int32
)

# === Model ===

encoder_input = layers.Input(
    shape=(sequence_length,), dtype="int32", name="question")
x = layers.Embedding(input_dim=max_tokens, output_dim=128)(encoder_input)
_, state_h, state_c = layers.LSTM(128, return_state=True)(x)

decoder_input = layers.Input(
    shape=(sequence_length,), dtype="int32", name="answer")
x = layers.Embedding(input_dim=max_tokens, output_dim=128)(decoder_input)
x, _, _ = layers.LSTM(128, return_sequences=True, return_state=True)(
    x, initial_state=[state_h, state_c])
decoder_output = layers.Dense(max_tokens, activation="softmax")(x)

model = Model([encoder_input, decoder_input], decoder_output)
model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.summary()

# === Train ===
model.fit(batched_dataset, epochs=10)
