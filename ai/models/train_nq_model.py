
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModel
import os

TRAIN_TFRECORD = "ai/datasets/cleaned_data.train.tfrecord"
VAL_TFRECORD = "ai/datasets/cleaned_data.val.tfrecord"
TOKENIZER_PATH = "ai/tokenizer_model"
BATCH_SIZE = 32
MAX_LENGTH = 75
VOCAB_SIZE = 30522  # Default for BERT base
EPOCHS = 5

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)

def _parse_function(example_proto):
    feature_description = {
        "question": tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
        "answer": tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
    }
    parsed = tf.io.parse_single_example(example_proto, feature_description)
    return parsed["question"], parsed["answer"]

def load_dataset(tfrecord_path):
    raw_dataset = tf.data.TFRecordDataset(tfrecord_path)
    parsed_dataset = raw_dataset.map(_parse_function)

    def format_batch(q, a):
        decoder_input = tf.concat([[tokenizer.cls_token_id], a[:-1]], axis=0)
        decoder_input = tf.pad(decoder_input, [[0, MAX_LENGTH - tf.shape(decoder_input)[0]]])[:MAX_LENGTH]
        target = tf.pad(a, [[0, MAX_LENGTH - tf.shape(a)[0]]])[:MAX_LENGTH]
        q = tf.pad(q, [[0, MAX_LENGTH - tf.shape(q)[0]]])[:MAX_LENGTH]
        return (q, decoder_input), target

    return (
        parsed_dataset
        .map(format_batch, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

train_ds = load_dataset(TRAIN_TFRECORD)
val_ds = load_dataset(VAL_TFRECORD)

# Encoder
encoder_input = tf.keras.layers.Input(shape=(MAX_LENGTH,), dtype="int32", name="encoder_input")
encoder_embed = tf.keras.layers.Embedding(VOCAB_SIZE, 128)(encoder_input)
_, state_h, state_c = tf.keras.layers.LSTM(128, return_state=True)(encoder_embed)

# Decoder
decoder_input = tf.keras.layers.Input(shape=(MAX_LENGTH,), dtype="int32", name="decoder_input")
decoder_embed = tf.keras.layers.Embedding(VOCAB_SIZE, 128)(decoder_input)
decoder_lstm = tf.keras.layers.LSTM(128, return_sequences=True)
decoder_output = decoder_lstm(decoder_embed, initial_state=[state_h, state_c])
output_layer = tf.keras.layers.Dense(VOCAB_SIZE, activation="softmax")(decoder_output)

model = tf.keras.Model([encoder_input, decoder_input], output_layer)
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Wrap target labels with extra dimension for sparse_categorical_crossentropy
def expand_targets(inputs, target):
    return inputs, tf.expand_dims(target, -1)

train_ds = train_ds.map(expand_targets)
val_ds = val_ds.map(expand_targets)

model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

model.save("ai/saved_models/trained_nq_model.keras")
print("✅ Model saved to trained_nq_model/")
