from pathlib import Path
from typing import List

import chromadb
from chromadb.utils import embedding_functions

CORPUS_DIR = Path("data/gst_corpus")
CHROMA_DIR = ".chroma"
COLLECTION_NAME = "gst_rules"

# use sentence-transformers locally — no API calls, no quota issues
_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)


def _chunk_text(text: str, chunk_size: int = 400, overlap: int = 80) -> List[str]:
    """Split text into overlapping chunks by character count."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end].strip())
        start += chunk_size - overlap
    return [c for c in chunks if len(c) > 50]  # drop tiny tail chunks


def build_corpus() -> chromadb.Collection:
    """
    Load all .txt files from gst_corpus/, chunk them,
    embed with sentence-transformers, and persist to ChromaDB.
    Returns the collection.
    """
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # wipe and rebuild so reruns are idempotent
    # wipe and rebuild so reruns are idempotent
    try:
        client.delete_collection(COLLECTION_NAME)
    except ValueError:
        pass
    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=_ef,
        metadata={"hnsw:space": "cosine"},
    )

    docs, ids, metadatas = [], [], []
    chunk_index = 0

    for txt_file in sorted(CORPUS_DIR.glob("*.txt")):
        text = txt_file.read_text(encoding="utf-8")
        chunks = _chunk_text(text)
        for chunk in chunks:
            docs.append(chunk)
            ids.append(f"chunk_{chunk_index}")
            metadatas.append({"source": txt_file.name})
            chunk_index += 1

    collection.add(documents=docs, ids=ids, metadatas=metadatas)
    print(f"✓ Indexed {chunk_index} chunks from {len(list(CORPUS_DIR.glob('*.txt')))} files")
    return collection


def get_collection() -> chromadb.Collection:
    """Load existing ChromaDB collection without rebuilding."""
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return client.get_collection(name=COLLECTION_NAME, embedding_function=_ef)