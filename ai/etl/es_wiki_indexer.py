#!/usr/bin/env python3
import os
import json
import glob
from elasticsearch import Elasticsearch, helpers
import argparse
import urllib3
from ..settings import settings

urllib3.disable_warnings()


def index_wiki_passages(wiki_dir, index_name="qa_passages"):
    # 1️⃣ Connect to Elasticsearch
    es = Elasticsearch(
         hosts=[
            {'host': settings.elastic_host,
             'port': settings.elastic_port,
             'scheme': 'https'}
        ],
        verify_certs=settings.elastic_veryify_cert,
        basic_auth=(settings.elastic_username, settings.elastic_password))

    # 2️⃣ Create index with BM25 mapping if it doesn't exist
    if not es.indices.exists(index=index_name):
        es.indices.create(
            index=index_name,
            body={
                "settings": {
                    "index": {
                        "max_ngram_diff": 2
                    },
                    "analysis": {
                        "tokenizer": {
                            "ngram_tokenizer": {
                                "type": "ngram",
                                "min_gram": 3,
                                "max_gram": 5,
                                "token_chars": ["letter", "digit"]
                            }
                        },
                        "analyzer": {
                            "ngram_analyzer": {
                                "tokenizer": "ngram_tokenizer",
                                "filter": ["lowercase"]
                            }
                        }
                    }
                },
                "mappings": {
                    "properties": {
                        "text": {
                            "type": "text",
                            "fields": {
                                "ngram": {
                                    "type": "text",
                                    "analyzer": "ngram_analyzer"
                                }
                            }
                        }
                    }
                }
            }
        )

    actions = []
    doc_id = 0

    # 3️⃣ Iterate over every file in wiki_dir recursively
    pattern = os.path.join(wiki_dir, "**", "*")
    for file_path in glob.glob(pattern, recursive=True):
        if not os.path.isfile(file_path):
            continue
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    # Parse each line as JSON
                    try:
                        article = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    # Extract text (some extracts use 'text', others 'payload')
                    text = article.get("text") or article.get("payload") or ""
                    if not text:
                        continue
                    # Split into paragraphs and index those ≥50 chars
                    for para in text.split("\n\n"):
                        para = para.strip()
                        if len(para) < 50:
                            continue
                        actions.append({
                            "_op_type": "index",
                            "_index": index_name,
                            "_id": doc_id,
                            "_source": {"text": para}
                        })
                        doc_id += 1
                        # Bulk-index in chunks of 1000
                        if len(actions) >= 1000:
                            helpers.bulk(es, actions)
                            actions = []
        except Exception as e:
            print(f"⚠️ Skipping file {file_path}: {e}")

    # 4️⃣ Flush any remaining actions
    if actions:
        helpers.bulk(es, actions)

    # 5️⃣ Refresh and report
    es.indices.refresh(index=index_name)
    print(f"✅ Indexed {doc_id} passages into '{index_name}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Index WikiExtractor JSONL output into Elasticsearch"
    )
    parser.add_argument(
        "--wiki_dir", type=str, default="ai/wiki_json",
        help="Directory containing WikiExtractor JSONL files (any extension)"
    )
    parser.add_argument(
        "--index", type=str, default="qa_passages",
        help="Name of the Elasticsearch index to create/use"
    )
    args = parser.parse_args()

    index_wiki_passages(
        wiki_dir=args.wiki_dir,
        index_name=args.index
    )
