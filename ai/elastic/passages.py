from elasticsearch import Elasticsearch, helpers

# TODO move to environment variables.
es_host="http://localhost:9200"
INDEX="qa_passages"

def retrieve(query, k=3):
    es = Elasticsearch(es_host)
    
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
print(retrieve("What is the capital of France?", k=2))
# → ["The capital of France is Paris, …", "…"]
