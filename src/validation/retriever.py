from typing import List, Dict
from src.validation.corpus_loader import get_collection


def retrieve_rules(query: str, n_results: int = 3) -> List[Dict]:
    """
    Retrieve relevant GST rules for a given query.
    Returns list of {text, source, distance} dicts.
    """
    collection = get_collection()
    results = collection.query(query_texts=[query], n_results=n_results)

    retrieved = []
    for doc, meta, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        retrieved.append({
            "text": doc,
            "source": meta["source"],
            "relevance_score": round(1 - distance, 4),
        })
    return retrieved