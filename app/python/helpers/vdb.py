from typing import List, Dict, Any, Optional
import faiss
import numpy as np
from app.models import get_embedding_model
import logging


class Document:
    id: str
    content: str
    metadata: Dict[str, Any]

    def __init__(
        self, document_id: str, content: str, metadata: Optional[Dict[str, Any]] = None
    ):
        self.id: str = document_id
        self.content: str = content
        self.metadata: Dict[str, Any] = metadata or {}


class VectorDB:
    def __init__(self, config: Dict[str, Any]):
        self.config: Dict[str, Any] = config
        self.embedding_model = get_embedding_model(config["embeddings_model"])
        self.index = faiss.IndexFlatL2(self.embedding_model.embedding_dim)
        self.documents: Dict[str, Document] = {}

    def add_documents(self, documents: List[Document]) -> None:
        for doc in documents:
            embedding = self.embedding_model.embed_query(doc.content)
            self.index.add(np.array([embedding]))
            self.documents[doc.id] = doc

    def search(self, query: str, top_k: int = 5) -> List[Document]:
        try:
            query_embedding = self.embedding_model.embed_query(query)
            _, indices = self.index.search(np.array([query_embedding]), top_k)
            return [
                self.documents[str(i)] for i in indices[0] if str(i) in self.documents
            ]
        except Exception as e:
            logging.error(f"Error during vector search: {str(e)}")
            return []

    def delete_document(self, document_id: str) -> None:
        if document_id in self.documents:
            del self.documents[document_id]
            # Note: FAISS doesn't support direct deletion,
            # so we need to rebuild the index
            self.rebuild_index()

    def update_document(
        self,
        document_id: str,
        new_content: str,
        new_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if document_id in self.documents:
            self.documents[document_id].content = new_content
            if new_metadata:
                self.documents[document_id].metadata = new_metadata
            self.rebuild_index()

    def rebuild_index(self):
        self.index = faiss.IndexFlatL2(self.embedding_model.embedding_dim)
        for doc in self.documents.values():
            embedding = self.embedding_model.embed_query(doc.content)
            self.index.add(np.array([embedding]))

    def search_similarity_threshold(
        self, query: str, top_k: int = 5, threshold: float = 0.1
    ) -> List[Document]:
        query_embedding = self.embedding_model.embed_query(query)
        distances, indices = self.index.search(np.array([query_embedding]), top_k)
        return [
            self.documents[str(idx)]
            for dist, idx in zip(distances[0], indices[0])
            if dist <= threshold and str(idx) in self.documents
        ]

    def insert_document(
        self, content: str, metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        doc_id = str(len(self.documents))
        doc = Document(doc_id, content, metadata)
        self.add_documents([doc])
        return doc_id

    def delete_documents_by_ids(self, ids: List[str]) -> int:
        deleted_count = 0
        for doc_id in ids:
            if doc_id in self.documents:
                del self.documents[doc_id]
                deleted_count += 1
        self.rebuild_index()
        return deleted_count

    def delete_documents_by_query(self, query: str) -> int:
        docs_to_delete = self.search(query)
        return self.delete_documents_by_ids([doc.id for doc in docs_to_delete])
