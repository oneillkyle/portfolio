import tensorflow as tf
from transformers import AutoTokenizer
import numpy as np
import os

TRAIN_TFRECORD = "ai/datasets/cleaned_data.train.tfrecord"
VAL_TFRECORD = "ai/datasets/cleaned_data.val.tfrecord"
TOKENIZER_PATH = "ai/tokenizer_model"
MODEL_SAVE_PATH = "ai/saved_models/trained_nq_model.keras"
MAX_LENGTH = 75
BATCH_SIZE = 32
EPOCHS = 10

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
SEP_ID = tokenizer.sep_token_id
CLS_ID = tokenizer.cls_token_id
PAD_ID = tokenizer.pad_token_id


def _parse_function(example_proto):
    features = {
        "question": tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
        "answer": tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
    }
    ex = tf.io.parse_single_example(example_proto, features)
    return ex["question"], ex["answer"]

def load_dataset(path):
    ds = tf.data.TFRecordDataset(path)
    ds = ds.map(_parse_function)

    def format(q, a):
        q = q[:MAX_LENGTH]
        a = a[:MAX_LENGTH - 1]  # Leave room for [SEP]
        a = tf.concat([a, [SEP_ID]], axis=0)
        decoder_input = tf.concat([[CLS_ID], a[:-1]], axis=0)

        q = tf.pad(q, [[0, MAX_LENGTH - tf.shape(q)[0]]])[:MAX_LENGTH]
        decoder_input = tf.pad(decoder_input, [[0, MAX_LENGTH - tf.shape(decoder_input)[0]]])[:MAX_LENGTH]
        target = tf.pad(a, [[0, MAX_LENGTH - tf.shape(a)[0]]])[:MAX_LENGTH]
        return (q, decoder_input), target

    ds = ds.map(format).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds

train_ds = load_dataset(TRAIN_TFRECORD)
val_ds = load_dataset(VAL_TFRECORD)

# Model
inp_q = tf.keras.Input(shape=(MAX_LENGTH,), dtype=tf.int32)
inp_a = tf.keras.Input(shape=(MAX_LENGTH,), dtype=tf.int32)
emb = tf.keras.layers.Embedding(30522, 128, mask_zero=True)
q_emb = emb(inp_q)
a_emb = emb(inp_a)
_, h, c = tf.keras.layers.LSTM(128, return_state=True)(q_emb)
dec_out = tf.keras.layers.LSTM(128, return_sequences=True)(a_emb, initial_state=[h, c])
logits = tf.keras.layers.Dense(30522, activation='softmax')(dec_out)
model = tf.keras.Model([inp_q, inp_a], logits)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

def expand_dims(inputs, target):
    return inputs, tf.expand_dims(target, -1)

train_ds = train_ds.map(expand_dims)
val_ds = val_ds.map(expand_dims)

# Debug preview
sample_batch = next(iter(train_ds))
print("\U0001F50D Sample Q:", tokenizer.decode(sample_batch[0][0][0].numpy(), skip_special_tokens=True))
print("\U0001F50D Target A:", tokenizer.decode(sample_batch[1][0][:MAX_LENGTH].numpy().flatten().tolist(), skip_special_tokens=True))

# Callbacks
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True)
early_stop_cb = tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)

model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=[checkpoint_cb, early_stop_cb])
model.save(MODEL_SAVE_PATH)
