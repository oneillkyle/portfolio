
import tensorflow as tf
from tensorflow.keras import layers, Model
from tfrecord_loader import load_tfrecord

# === Load data ===
sequence_length = 50
batch_size = 32
dataset = load_tfrecord("ai/datasets/cleaned_data.tfrecord", sequence_length=sequence_length, batch_size=batch_size)

# === Define model ===
vocab_size = 10000  # Must match the tokenizer's vocab size

encoder_input = layers.Input(shape=(sequence_length,), dtype="int32", name="encoder_input")
encoder_emb = layers.Embedding(input_dim=vocab_size, output_dim=128, mask_zero=True)(encoder_input)
_, state_h, state_c = layers.LSTM(128, return_state=True)(encoder_emb)

decoder_input = layers.Input(shape=(sequence_length,), dtype="int32", name="decoder_input")
decoder_emb = layers.Embedding(input_dim=vocab_size, output_dim=128, mask_zero=True)(decoder_input)
decoder_lstm, _, _ = layers.LSTM(128, return_sequences=True, return_state=True)(decoder_emb, initial_state=[state_h, state_c])
decoder_output = layers.Dense(vocab_size, activation="softmax")(decoder_lstm)

model = Model(inputs=[encoder_input, decoder_input], outputs=decoder_output)

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# === Train ===
model.fit(dataset, epochs=10)

# === Save model ===
model.save("api/app/saved_models/nq_model")
print("✅ Model saved to 'api/app/saved_models/nq_model'")
