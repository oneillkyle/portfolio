#!/usr/bin/env python3
import os
import glob
import json
import argparse

def build_wiki_corpus(wiki_dir: str, output_file: str, min_len: int = 50):
    """
    Reads all files under `wiki_dir` recursively, parses each line as JSON,
    splits the "text" field into paragraphs on blank lines, and writes
    paragraphs of length > min_len to output_file (one per line).
    """
    count_in, count_out = 0, 0
    with open(output_file, "w", encoding="utf-8") as out:
        # Walk through every path under wiki_dir
        pattern = os.path.join(wiki_dir, "**", "*")
        for fn in glob.glob(pattern, recursive=True):
            if not os.path.isfile(fn):
                continue
            try:
                with open(fn, "r", encoding="utf-8") as f:
                    for line in f:
                        count_in += 1
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        text = obj.get("text") or obj.get("payload") or ""
                        if not text:
                            continue
                        # Split into paragraphs
                        for para in text.split("\n\n"):
                            p = para.strip().replace("\n", " ")
                            if len(p) > min_len:
                                out.write(p + "\n")
                                count_out += 1
            except Exception as e:
                print(f"⚠️ Skipping file {fn}: {e}")
    print(f"Read {count_in} lines; wrote {count_out} paragraphs to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Flatten WikiExtractor JSONL folder into a single corpus text file."
    )
    parser.add_argument(
        "--wiki_dir",
        type=str,
        default="ai/wiki_json",
        help="Path to the root folder of WikiExtractor JSONL files"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="ai/datasets/wiki_corpus.txt",
        help="Path for the output corpus text file"
    )
    parser.add_argument(
        "--min_len",
        type=int,
        default=50,
        help="Minimum paragraph length (characters) to include"
    )
    args = parser.parse_args()

    build_wiki_corpus(args.wiki_dir, args.output, args.min_len)
