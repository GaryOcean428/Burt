from typing import List, Dict, Any
import uuid

class Document:
    def __init__(self, content: str, metadata: Dict[str, Any] = None):
        self.content = content
        self.metadata = metadata or {}

class VectorDB:
    def __init__(self, embeddings_model=None, in_memory=True, cache_dir=None):
        self.vectors = {}
        self.documents = {}
        self.embeddings_model = embeddings_model
        self.in_memory = in_memory
        self.cache_dir = cache_dir

    def add_document(self, key: str, document: Document, vector: List[float]):
        self.vectors[key] = vector
        self.documents[key] = document

    def get_document(self, key: str) -> Document:
        return self.documents.get(key)

    def get_vector(self, key: str) -> List[float]:
        return self.vectors.get(key, [])

    def remove_document(self, key: str):
        if key in self.vectors:
            del self.vectors[key]
        if key in self.documents:
            del self.documents[key]

    def search_similar(self, query_vector: List[float], k: int = 5) -> List[Dict[str, Any]]:
        similarities = []
        for key, vector in self.vectors.items():
            similarity = self.cosine_similarity(query_vector, vector)
            similarities.append({"key": key, "similarity": similarity, "document": self.documents[key]})

        similarities.sort(key=lambda x: x["similarity"], reverse=True)
        return similarities[:k]

    def search_similarity_threshold(self, query: str, count: int = 5, threshold: float = 0.1) -> List[Dict[str, Any]]:
        query_vector = self.embeddings_model.embed_query(query) if self.embeddings_model else []
        similarities = self.search_similar(query_vector, len(self.vectors))
        return [s for s in similarities[:count] if s["similarity"] >= threshold]

    def insert_document(self, text: str, metadata: Dict[str, Any] = None) -> str:
        doc_id = str(uuid.uuid4())
        document = Document(text, metadata)
        vector = self.embeddings_model.embed_query(text) if self.embeddings_model else []
        self.add_document(doc_id, document, vector)
        return doc_id

    def delete_documents_by_ids(self, ids: List[str]) -> int:
        deleted_count = 0
        for doc_id in ids:
            if doc_id in self.documents:
                self.remove_document(doc_id)
                deleted_count += 1
        return deleted_count

    def delete_documents_by_query(self, query: str) -> int:
        query_vector = self.embeddings_model.embed_query(query) if self.embeddings_model else []
        similarities = self.search_similar(query_vector, len(self.vectors))
        deleted_count = 0
        for similarity in similarities:
            if similarity["similarity"] > 0.9:  # Arbitrary threshold, adjust as needed
                self.remove_document(similarity["key"])
                deleted_count += 1
        return deleted_count

    @staticmethod
    def cosine_similarity(v1: List[float], v2: List[float]) -> float:
        dot_product = sum(a * b for a, b in zip(v1, v2))
        magnitude1 = sum(a * a for a in v1) ** 0.5
        magnitude2 = sum(b * b for b in v2) ** 0.5
        if magnitude1 * magnitude2 == 0:
            return 0
        return dot_product / (magnitude1 * magnitude2)
