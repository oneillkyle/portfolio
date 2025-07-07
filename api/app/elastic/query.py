from elasticsearch import Elasticsearch, helpers
from settings import settings
import urllib3

urllib3.disable_warnings()

INDEX = "qa_passages"


def retrieve(query, k=5):
    es = Elasticsearch(
        hosts=[
            {'host': settings.elastic_host,
             'port': settings.elastic_port,
             'scheme': 'https'}
        ],
        verify_certs=False,
        basic_auth=(settings.elastic_username, settings.elastic_password))

    resp = es.search(
        index=INDEX,
        body={
            "size": k,
            "query": {
                "bool": {
                    "must": [
                        {
                            "multi_match": {
                                "query": query,
                                "fields": ["text", "text.ngram"],
                                "type": "best_fields",
                                "operator": "and",
                                "minimum_should_match": "75%"
                            }
                        }
                    ],
                    "should": [
                        {
                            "match_phrase": {
                                "text": {
                                    "query": query,
                                    "boost": 2.0
                                }
                            }
                        }
                    ]
                }
            }
        }
    )
    return [hit["_source"]["text"] for hit in resp["hits"]["hits"]]


# Quick test
# print(retrieve("What is the capital of France?", k=2))
# → ["The capital of France is Paris, …", "…"]
