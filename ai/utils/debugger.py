import tensorflow as tf


# === Utility: Shape + dtype validation ===
def validate_tensor(tensor, *, expected_rank=None, expected_shape=None, expected_dtype=None, label=""):
    if expected_rank is not None:
        tf.debugging.assert_rank(
            tensor,
            expected_rank,
            message=f"{label} expected rank {expected_rank}, got shape {tensor.shape}"
        )
    if expected_shape is not None:
        tf.debugging.assert_shapes([
            (tensor, expected_shape)
        ], message=f"{label} expected shape {expected_shape}, got {tensor.shape}")
    if expected_dtype is not None:
        tf.debugging.assert_type(
            tensor,
            expected_dtype,
            message=f"{label} expected dtype {expected_dtype}, got {tensor.dtype}"
        )


def debug_dataset_shapes(
    dataset,
    name="dataset",
    num_samples=1,
    expected_input_shape=None,
    expected_input_dtype=None,
    expected_label_shape=None,
    expected_label_dtype=None,
):
    print(f"\n🧪 Inspecting '{name}'...")
    for i, sample in enumerate(dataset.take(num_samples)):
        print(f"\n🔹 Sample {i + 1}")

        if isinstance(sample, tuple):
            inputs, labels = sample

            if isinstance(inputs, tuple):
                for j, inp in enumerate(inputs):
                    print(
                        f"  🔸 Input {j + 1}: shape = {inp.shape}, dtype = {inp.dtype}")
                    validate_tensor(
                        inp,
                        expected_rank=len(
                            expected_input_shape) if expected_input_shape else None,
                        expected_shape=expected_input_shape,
                        expected_dtype=expected_input_dtype,
                        label=f"Input {j + 1}"
                    )
            else:
                print(
                    f"  🔸 Input: shape = {inputs.shape}, dtype = {inputs.dtype}")
                validate_tensor(
                    inputs,
                    expected_rank=len(
                        expected_input_shape) if expected_input_shape else None,
                    expected_shape=expected_input_shape,
                    expected_dtype=expected_input_dtype,
                    label="Input"
                )

            print(f"  🔸 Label: shape = {labels.shape}, dtype = {labels.dtype}")
            validate_tensor(
                labels,
                expected_rank=len(
                    expected_label_shape) if expected_label_shape else None,
                expected_shape=expected_label_shape,
                expected_dtype=expected_label_dtype,
                label="Label"
            )

        else:
            print(
                f"  🔸 Output: shape = {sample.shape}, dtype = {sample.dtype}")
            validate_tensor(
                sample,
                expected_rank=len(
                    expected_label_shape) if expected_label_shape else None,
                expected_shape=expected_label_shape,
                expected_dtype=expected_label_dtype,
                label="Output"
            )
