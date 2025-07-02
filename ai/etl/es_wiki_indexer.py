#!/usr/bin/env python3
import os
import json
import glob
from elasticsearch import Elasticsearch, helpers
import argparse

def index_wiki_passages(wiki_dir, es_host="http://localhost:9200", index_name="qa_passages"):
    # 1) Connect
    es = Elasticsearch(es_host)
    # 2) Create index with BM25 mapping if it doesn't exist
    if not es.indices.exists(index=index_name):
        es.indices.create(
            index=index_name,
            body={
                "mappings": {
                    "properties": {
                        "text": {"type": "text"}
                    }
                }
            }
        )

    # 3) Iterate over WikiExtractor JSON files
    actions = []
    doc_id = 0
    for json_file in glob.glob(os.path.join(wiki_dir, "**", "*.json"), recursive=True):
        with open(json_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    article = json.loads(line)
                    article_text = article.get("text", "")
                    # Split into paragraphs on blank lines
                    for para in article_text.split("\n\n"):
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
                        # Bulk in chunks of 1000
                        if len(actions) >= 1000:
                            helpers.bulk(es, actions)
                            actions = []
                except json.JSONDecodeError:
                    continue

    # 4) Flush remaining actions
    if actions:
        helpers.bulk(es, actions)
    es.indices.refresh(index=index_name)
    print(f"✅ Indexed {doc_id} passages into '{index_name}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Index WikiExtractor JSON output into Elasticsearch")
    parser.add_argument("--wiki_dir", type=str, required=True, default="ai/wiki_json",
                        help="Path to directory produced by WikiExtractor.py --json")
    parser.add_argument("--es_host", type=str, default="http://localhost:9200",
                        help="Elasticsearch host URL")
    parser.add_argument("--index", type=str, default="qa_passages",
                        help="Name of the Elasticsearch index")
    args = parser.parse_args()

    index_wiki_passages(
        wiki_dir=args.wiki_dir,
        es_host=args.es_host,
        index_name=args.index
    )
