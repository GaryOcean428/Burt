import unittest
from unittest.mock import MagicMock, patch
from app.python.helpers.vdb import VectorDB, Document
import numpy as np


class TestVectorDB(unittest.TestCase):
    def setUp(self):
        self.config = {"embeddings_model": "test-embeddings-model"}
        self.mock_embedding_model = MagicMock()
        self.mock_embedding_model.embedding_dim = 5
        self.mock_embedding_model.embed_query.return_value = np.array(
            [1, 2, 3, 4, 5]
        )

        with patch(
            "app.python.helpers.vdb.get_embedding_model",
            return_value=self.mock_embedding_model,
        ):
            self.vdb = VectorDB(self.config)

    def test_add_documents(self):
        docs = [
            Document("1", "Test content 1"),
            Document("2", "Test content 2"),
        ]
        self.vdb.add_documents(docs)
        self.assertEqual(len(self.vdb.documents), 2)
        self.assertIn("1", self.vdb.documents)
        self.assertIn("2", self.vdb.documents)

    def test_search(self):
        docs = [
            Document("0", "Test content 1"),
            Document("1", "Test content 2"),
        ]
        self.vdb.add_documents(docs)

        # Mock the FAISS index search method
        self.vdb.index.search = MagicMock(
            return_value=(None, np.array([[0, 1]]))
        )

        results = self.vdb.search("test query")
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].id, "0")
        self.assertEqual(results[1].id, "1")

    def test_delete_document(self):
        docs = [
            Document("1", "Test content 1"),
            Document("2", "Test content 2"),
        ]
        self.vdb.add_documents(docs)
        self.vdb.delete_document("1")
        self.assertEqual(len(self.vdb.documents), 1)
        self.assertNotIn("1", self.vdb.documents)
        self.assertIn("2", self.vdb.documents)

    def test_update_document(self):
        doc = Document("1", "Test content 1")
        self.vdb.add_documents([doc])
        self.vdb.update_document("1", "Updated content", {"key": "value"})
        updated_doc = self.vdb.documents["1"]
        self.assertEqual(updated_doc.content, "Updated content")
        self.assertEqual(updated_doc.metadata, {"key": "value"})

    def test_search_similarity_threshold(self):
        docs = [
            Document("0", "Test content 1"),
            Document("1", "Test content 2"),
        ]
        self.vdb.add_documents(docs)

        # Mock the FAISS index search method
        self.vdb.index.search = MagicMock(
            return_value=(np.array([[0.05, 0.15]]), np.array([[0, 1]]))
        )

        results = self.vdb.search_similarity_threshold(
            "test query", threshold=0.1
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].id, "0")


if __name__ == "__main__":
    unittest.main()
