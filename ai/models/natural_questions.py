import tensorflow as tf
import json
from tensorflow.keras.layers import TextVectorization

file_path = 'ai/datasets/simplified-nq-train.jsonl'
# dict_keys(['document_text', 'long_answer_candidates', 'question_text', 'annotations', 'document_url', 'example_id'])
# l['annotations'][0]['long_answer']
# {'start_token': 212, 'candidate_index': 15, 'end_token': 310}
# l['annotations'][0]['short_answers']
# [{'start_token': 213, 'end_token': 215}]

# Function to parse each line of the JSONL file
def parse_json_line(line):
    data = json.loads(line.numpy().decode("utf-8"))
    question = data.get("question_text", "")
    document_text = data.get("document_text", "")
    
    # Extract the short answer using start_token and end_token
    short_answer = data.get("annotations", [{}])[0].get("short_answers", [{}])[0]
    start_token = short_answer.get("start_token", None)
    end_token = short_answer.get("end_token", None)
    
    if start_token is not None and end_token is not None:
        # Extract the answer text from the document_text
        answer = " ".join(document_text.split()[start_token:end_token])
    else:
        answer = ""  # Default to an empty string if no valid tokens are found
    
    return question, answer

# Wrap the parsing function for TensorFlow compatibility
def tf_parse_json_line(line):
    question, answer = tf.py_function(func=parse_json_line, inp=[line], Tout=[tf.string, tf.string])
    return question, answer

# Load and preprocess the dataset
def load_jsonl_dataset(file_path):
    dataset = tf.data.TextLineDataset(file_path)
    dataset = dataset.map(tf_parse_json_line)
    dataset = dataset.batch(32).prefetch(tf.data.AUTOTUNE)
    return dataset

dataset = load_jsonl_dataset(file_path)

# Create TextVectorization layers for questions and answers
question_vectorizer = TextVectorization(max_tokens=10000, output_sequence_length=50)
answer_vectorizer = TextVectorization(max_tokens=10000, output_sequence_length=50)

# Adapt the vectorizers to the dataset
questions = dataset.map(lambda q, a: q)
answers = dataset.map(lambda q, a: a)
question_vectorizer.adapt(questions)
answer_vectorizer.adapt(answers)

# Vectorize the dataset
def vectorize_text(question, answer):
    question = question_vectorizer(question)
    answer = answer_vectorizer(answer)
    return question, answer

vectorized_dataset = dataset.map(vectorize_text)

from tensorflow.keras import layers, Model

# Define the encoder
encoder_input = layers.Input(shape=(50,), dtype="int64", name="question")
encoder_embedding = layers.Embedding(input_dim=10000, output_dim=128)(encoder_input)
encoder_output = layers.LSTM(128, return_state=True)
_, state_h, state_c = encoder_output(encoder_embedding)

# Define the decoder
decoder_input = layers.Input(shape=(50,), dtype="int64", name="answer")
decoder_embedding = layers.Embedding(input_dim=10000, output_dim=128)(decoder_input)
decoder_lstm = layers.LSTM(128, return_sequences=True, return_state=True)
decoder_output, _, _ = decoder_lstm(decoder_embedding, initial_state=[state_h, state_c])
decoder_dense = layers.Dense(10000, activation="softmax")
decoder_output = decoder_dense(decoder_output)

# Create the model
model = Model([encoder_input, decoder_input], decoder_output)
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

def prepare_training_data(question, answer):
    decoder_input = tf.concat([tf.constant([1]), answer[:-1]], axis=0)  # Add start token
    decoder_target = answer  # Target is the original answer
    return (question, decoder_input), decoder_target

training_data = vectorized_dataset.map(prepare_training_data)

model.fit(training_data, epochs=10)