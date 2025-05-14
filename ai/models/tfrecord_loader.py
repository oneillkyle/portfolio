import tensorflow as tf

# Parse the TFRecord into tensors


def parse_tfrecord(example_proto):
    feature_description = {
        "question": tf.io.VarLenFeature(tf.int64),
        "answer": tf.io.VarLenFeature(tf.int64),
    }
    parsed = tf.io.parse_single_example(example_proto, feature_description)
    q = tf.sparse.to_dense(parsed["question"])
    a = tf.sparse.to_dense(parsed["answer"])
    return q, a

# Load and batch the dataset


def load_tfrecord(path, sequence_length=50, batch_size=32):
    def pad_sequence(q, a):
        q = q[:sequence_length]
        a = a[:sequence_length]
        q = tf.pad(q, [[0, sequence_length - tf.shape(q)[0]]])
        a = tf.pad(a, [[0, sequence_length - tf.shape(a)[0]]])
        return (q, a[:-1]), a  # decoder_input, target

    dataset = tf.data.TFRecordDataset(path)
    dataset = dataset.map(parse_tfrecord)
    dataset = dataset.map(pad_sequence)
    dataset = dataset.shuffle(1000).batch(
        batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset
