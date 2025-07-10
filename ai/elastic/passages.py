from elasticsearch import Elasticsearch, helpers
from ..settings import settings

INDEX = "qa_passages"


def retrieve(query, k=3):
    es = Elasticsearch(
        hosts=[
            {'host': settings.elastic_host,
             'port': settings.elastic_port,
             'scheme': 'https'}
        ],
        verify_certs=settings.elastic_veryify_cert,
        basic_auth=(settings.elastic_username, settings.elastic_password))

    resp = es.search(
        index=INDEX,
        body={
            "size": k,
            "query": {
                "match": {
                    "text": {
                        "query": query,
                        "operator": "and"
                    }
                }
            }
        }
    )
    return [hit["_source"]["text"] for hit in resp["hits"]["hits"]]


# Quick test
# print(retrieve("What is the capital of France?", k=2))
# → ["The capital of France is Paris, …", "…"]
