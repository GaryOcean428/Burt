from pinecone import Pinecone
import os
from typing import List, Dict, Any

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

index_name = os.getenv("PINECONE_INDEX_NAME")
index = pc.Index(index_name)

def upsert_vectors(vectors: List[Dict[str, Any]]):
    return index.upsert(vectors=vectors)

def query_vectors(query_vector: List[float], top_k: int = 5):
    return index.query(vector=query_vector, top_k=top_k)

def delete_vectors(ids: List[str]):
    return index.delete(ids=ids)
