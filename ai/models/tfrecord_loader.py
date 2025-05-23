
import tensorflow as tf

start_token = 1
sequence_length = 75  # Increased from 50

def parse_tfrecord(example_proto):
    feature_description = {
        "question": tf.io.VarLenFeature(tf.int64),
        "answer": tf.io.VarLenFeature(tf.int64),
    }
    parsed = tf.io.parse_single_example(example_proto, feature_description)
    q = tf.sparse.to_dense(parsed["question"])
    a = tf.sparse.to_dense(parsed["answer"])
    return q, a

def load_tfrecord(path, sequence_length=75, batch_size=32):
    def pad_sequence(q, a):
        q = q[:sequence_length]
        a = a[:sequence_length]

        q = tf.pad(q, [[0, sequence_length - tf.shape(q)[0]]])
        a = tf.pad(a, [[0, sequence_length - tf.shape(a)[0]]])

        decoder_input = tf.concat([[start_token], a[:-1]], axis=0)
        decoder_input = decoder_input[:sequence_length]
        decoder_input = tf.pad(decoder_input, [[0, sequence_length - tf.shape(decoder_input)[0]]])

        return (q, decoder_input), a

    dataset = tf.data.TFRecordDataset(path)
    dataset = dataset.map(parse_tfrecord)
    dataset = dataset.map(pad_sequence)
    dataset = dataset.shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset
