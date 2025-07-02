import argparse
import json
import random

def split_jsonl(input_path, train_path, val_path, train_ratio, seed):
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = [line for line in f if line.strip()]
    random.Random(seed).shuffle(lines)
    split_idx = int(len(lines) * train_ratio)
    train_lines = lines[:split_idx]
    val_lines = lines[split_idx:]

    with open(train_path, 'w', encoding='utf-8') as f:
        for line in train_lines:
            f.write(line)
    with open(val_path, 'w', encoding='utf-8') as f:
        for line in val_lines:
            f.write(line)
    print(f"Total examples: {len(lines)}")
    print(f"Train examples ({train_ratio*100}%): {len(train_lines)} -> {train_path}")
    print(f"Validation examples ({(1-train_ratio)*100}%): {len(val_lines)} -> {val_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split a JSONL file into train and validation sets.")
    parser.add_argument("--input", type=str, default="ai/datasets/cleaned_nq.jsonl", help="Input JSONL file")
    parser.add_argument("--train_out", type=str, default="ai/datasets/cleaned_nq.train.jsonl", help="Output training JSONL file")
    parser.add_argument("--val_out", type=str, default="ai/datasets/cleaned_nq.val.jsonl", help="Output validation JSONL file")
    parser.add_argument("--train_ratio", type=float, default=0.9, help="Proportion of data to use for training")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling")
    args = parser.parse_args()

    split_jsonl(args.input, args.train_out, args.val_out, args.train_ratio, args.seed)