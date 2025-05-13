import tensorflow as tf
import json
from tensorflow.keras.layers import TextVectorization

# === Load and parse the dataset ===
file_path = 'ai/datasets/simplified-nq-train.jsonl'

# Parse each JSONL line to extract question and short answer span
def parse_json_line(line):
    data = json.loads(line.numpy().decode("utf-8"))
    question = data.get("question_text", "")
    document_text = data.get("document_text", "")
    
    short_answers = data.get("annotations", [{}])[0].get("short_answers", [])
    answer = ""
    if short_answers:
        short_answer = short_answers[0]
        start_token = short_answer.get("start_token")
        end_token = short_answer.get("end_token")
        if start_token is not None and end_token is not None:
            answer = " ".join(document_text.split()[start_token:end_token])
    
    return question, answer

# Wrap in a TensorFlow-compatible function
def tf_parse_json_line(line):
    question, answer = tf.py_function(parse_json_line, [line], [tf.string, tf.string])
    question.set_shape([])
    answer.set_shape([])
    return question, answer

# Load JSONL and parse it
def load_jsonl_dataset(file_path):
    dataset = tf.data.TextLineDataset(file_path)
    dataset = dataset.map(tf_parse_json_line)
    return dataset

dataset = load_jsonl_dataset(file_path)

# === Text vectorization ===
max_tokens = 10000
sequence_length = 50

question_vectorizer = TextVectorization(max_tokens=max_tokens, output_sequence_length=sequence_length)
answer_vectorizer = TextVectorization(max_tokens=max_tokens, output_sequence_length=sequence_length)

# Adapt vectorizers
questions = dataset.map(lambda q, a: q).batch(256)
answers = dataset.map(lambda q, a: a).batch(256)
question_vectorizer.adapt(questions)
answer_vectorizer.adapt(answers)

# Vectorize the dataset
def vectorize_text(question, answer):
    return question_vectorizer(question), answer_vectorizer(answer)

vectorized_dataset = dataset.map(vectorize_text)

# === Add decoder inputs and targets ===
START_TOKEN = 1  # Token ID to represent "start of sequence"

def add_decoder_inputs_and_targets(question, answer):
    start_tokens = tf.fill([tf.shape(answer)[0], 1], START_TOKEN)
    decoder_input = tf.concat([start_tokens, answer[:, :-1]], axis=1)
    return (question, decoder_input), answer

# Batch the dataset before preparing decoder inputs
vectorized_dataset = vectorized_dataset.batch(32).map(add_decoder_inputs_and_targets)
vectorized_dataset = vectorized_dataset.prefetch(tf.data.AUTOTUNE)

# === Build the encoder-decoder model ===
from tensorflow.keras import layers, Model

# Encoder
encoder_input = layers.Input(shape=(sequence_length,), dtype="int32", name="question")
x = layers.Embedding(input_dim=max_tokens, output_dim=128)(encoder_input)
_, state_h, state_c = layers.LSTM(128, return_state=True)(x)

# Decoder
decoder_input = layers.Input(shape=(sequence_length,), dtype="int32", name="answer")
x = layers.Embedding(input_dim=max_tokens, output_dim=128)(decoder_input)
x, _, _ = layers.LSTM(128, return_sequences=True, return_state=True)(x, initial_state=[state_h, state_c])
decoder_output = layers.Dense(max_tokens, activation="softmax")(x)

# Final model
model = Model([encoder_input, decoder_input], decoder_output)
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# === Train ===
model.fit(vectorized_dataset, epochs=10)
