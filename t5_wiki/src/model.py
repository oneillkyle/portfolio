from __future__ import annotations
from typing import Union

import tensorflow as tf
from transformers import TFAutoModelForSeq2SeqLM, AutoConfig


class HFSeq2SeqWrapper(tf.keras.Model):
    def __init__(self, base_model: TFAutoModelForSeq2SeqLM):
        super().__init__()
        self.base = base_model

    def call(self, input_ids: tf.Tensor, training: bool = False) -> tf.Tensor:
        outputs = self.base(input_ids=input_ids, training=training)
        return outputs.logits

    # passthrough save to allow HF-compatible checkpoints
    def save_pretrained(self, save_directory: str) -> None:
        self.base.save_pretrained(save_directory)


def build_model(model_name_or_path: Union[str, bytes]) -> tf.keras.Model:
    config = AutoConfig.from_pretrained(model_name_or_path)
    base = TFAutoModelForSeq2SeqLM.from_pretrained(model_name_or_path, config=config)
    return HFSeq2SeqWrapper(base)
