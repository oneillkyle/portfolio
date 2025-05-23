

import tensorflow as tf
from tensorflow.keras import layers, Model
from tfrecord_loader import load_tfrecord
from tokenizers import Tokenizer

sequence_length = 75
batch_size = 32
dataset = load_tfrecord("ai/datasets/cleaned_data.tfrecord", sequence_length=sequence_length, batch_size=batch_size)

tokenizer = Tokenizer.from_file("ai/tokenizers/nq_tokenizer.json")
vocab_size = tokenizer.get_vocab_size()
start_token = tokenizer.token_to_id("[START]")
end_token = tokenizer.token_to_id("[END]")

encoder_input = layers.Input(shape=(sequence_length,), dtype="int32", name="encoder_input")
x = layers.Embedding(input_dim=vocab_size, output_dim=256, mask_zero=True)(encoder_input)
x = layers.LSTM(256, return_sequences=True)(x)
_, state_h, state_c = layers.LSTM(256, return_state=True)(x)

decoder_input = layers.Input(shape=(sequence_length,), dtype="int32", name="decoder_input")
y = layers.Embedding(input_dim=vocab_size, output_dim=256, mask_zero=True)(decoder_input)
y = layers.LSTM(256, return_sequences=True)(y, initial_state=[state_h, state_c])
decoder_output = layers.Dense(vocab_size, activation="softmax")(y)

model = Model(inputs=[encoder_input, decoder_input], outputs=decoder_output)
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.summary()

def generate_answer(model, tokenizer, question, sequence_length=75):
    ids = tokenizer.encode(question).ids
    ids = [start_token] + ids + [end_token]
    ids = ids[:sequence_length]
    ids += [0] * (sequence_length - len(ids))

    encoder_input = tf.convert_to_tensor([ids], dtype=tf.int32)
    decoder_input = tf.convert_to_tensor([[start_token] + [0] * (sequence_length - 1)], dtype=tf.int32)

    for i in range(1, sequence_length):
        preds = model([encoder_input, decoder_input], training=False)
        pred_id = tf.argmax(preds[0, i - 1]).numpy()
        if pred_id == end_token:
            break
        decoder_input = tf.tensor_scatter_nd_update(decoder_input, [[0, i]], [pred_id])

    return tokenizer.decode(decoder_input[0].numpy()[1:i])

class ShowPredictions(tf.keras.callbacks.Callback):
    def __init__(self, tokenizer, sample_questions, sequence_length=75):
        self.tokenizer = tokenizer
        self.sample_questions = sample_questions
        self.sequence_length = sequence_length

    def on_epoch_end(self, epoch, logs=None):
        print(f"\n📢 Epoch {epoch + 1} predictions:")
        for q in self.sample_questions:
            a = generate_answer(self.model, self.tokenizer, q, self.sequence_length)
            print("Q:", q)
            print("A:", a)
            print("-" * 30)

sample_questions = [
    "What is the capital of the United States?",
    "When did the Titanic sink?",
    "Where is Mount Everest located?",
    "Who painted the Mona Lisa?",
]

callbacks = [ShowPredictions(tokenizer, sample_questions)]

# === Train ===
history = model.fit(dataset, epochs=10, callbacks=callbacks)
model.save("ai/saved_models/nq_model.keras")

plt.plot(history.history["loss"])
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()
