
import tensorflow as tf

model = tf.keras.models.load_model("ai/saved_models/trained_nq_model.keras")
model.save("export_hf_format", save_format="tf")
print("✅ Saved model to export_hf_format/")
