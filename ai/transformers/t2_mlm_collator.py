import random
from dataclasses import dataclass
from typing import List, Dict, Any, Union
import numpy as np
import torch
from transformers import PreTrainedTokenizerBase

@dataclass
class DataCollatorForT5MLM:
    """
    Data collator used for T5 span‐masking (denoising) pretraining.
    Masks spans of tokens at random, replacing them with sentinel tokens.
    Implementation lifted from:
    https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/data_collator.py
    """
    tokenizer: PreTrainedTokenizerBase
    noise_density: float = 0.15
    mean_noise_span_length: int = 3
    input_length: int = 512
    target_length: int = 512

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Convert list of dicts to batch of input_ids
        input_ids = [e["input_ids"] for e in examples]
        batch_input = torch.tensor(self._pad(input_ids, self.tokenizer.pad_token_id), dtype=torch.long)
        batch_size, seq_length = batch_input.shape

        # Determine number of mask tokens
        mask_indices = self.random_spans_noise_mask(batch_size, seq_length)
        sentinel_ids = self.create_sentinel_ids(mask_indices)

        # Build model inputs
        inputs = batch_input.clone()
        inputs[mask_indices] = sentinel_ids[mask_indices]

        # Build labels: everything not masked is ignored (-100), masked tokens are replaced with sentinel ids
        labels = torch.full_like(batch_input, fill_value=-100)
        labels[mask_indices] = batch_input[mask_indices]

        # Truncate/pad to fixed lengths
        inputs = inputs[:, : self.input_length]
        labels = labels[:, : self.target_length]

        return {"input_ids": inputs, "labels": labels}

    def random_spans_noise_mask(self, batch_size, seq_length):
        """Generate mask for spans to be replaced by sentinel tokens."""
        num_noise_tokens = int(round(seq_length * self.noise_density))
        # Draw span lengths from Poisson
        span_lengths = np.random.poisson(self.mean_noise_span_length, size=(batch_size,))
        span_lengths = np.clip(span_lengths, 1, seq_length)
        # Create mask per example
        mask = torch.zeros((batch_size, seq_length), dtype=torch.bool)
        for i in range(batch_size):
            num_to_mask = num_noise_tokens
            spans = []
            while num_to_mask > 0:
                length = min(span_lengths[i], num_to_mask)
                start = random.randrange(0, seq_length - length + 1)
                spans.append((start, start + length))
                num_to_mask -= length
            for (s, e) in spans:
                mask[i, s:e] = True
        return mask

    def create_sentinel_ids(self, mask_indices):
        """Replace each masked span with a unique sentinel token."""
        batch_size, seq_length = mask_indices.shape
        sentinel_ids = torch.full((batch_size, seq_length), fill_value=self.tokenizer.pad_token_id)
        for i in range(batch_size):
            spans = self.masked_spans(mask_indices[i])
            for idx, (s, e) in enumerate(spans):
                sentinel_token_id = self.tokenizer.convert_tokens_to_ids(f"<extra_id_{idx}>")
                sentinel_ids[i, s] = sentinel_token_id
        return sentinel_ids

    def masked_spans(self, mask_row: torch.Tensor):
        """Convert a 1D mask tensor to list of (start, end) spans."""
        spans = []
        in_span = False
        for idx, m in enumerate(mask_row.tolist()):
            if m and not in_span:
                start = idx
                in_span = True
            if (not m or idx == len(mask_row) - 1) and in_span:
                end = idx if not m else idx + 1
                spans.append((start, end))
                in_span = False
        return spans

    def _pad(self, list_of_lists, pad_id):
        """Simple left‐and‐right pad to equal lengths."""
        max_len = max(len(x) for x in list_of_lists)
        return [x + [pad_id] * (max_len - len(x)) for x in list_of_lists]
