import unittest
import asyncio
import os
import json
import time
import random
from app.advanced_router import AdvancedRouter
from app.agent import Agent, AgentConfig
from app.python.helpers.rag_system import RAGSystem
from app.python.helpers.pinecone_db import (
    upsert_vectors,
    query_vectors,
    delete_vectors,
    check_pinecone_health,
)
from app.python.helpers.redis_cache import RedisCache
from app.python.helpers.mongodb_client import (
    insert_document,
    find_documents,
    update_document,
    delete_document,
    check_mongodb_health,
)
from app.python.helpers.perplexity_search import perplexity_search, assess_complexity
from app.config import load_config


class TestBurtonAISystem(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        config = load_config()
        agent_config = AgentConfig(**config)
        cls.agent = Agent(1, agent_config)
        cls.rag_system = RAGSystem()
        cls.router = AdvancedRouter(config, cls.agent, cls.rag_system)

    def test_advanced_router(self):
        query = "What is the capital of France?"
        complexity = self.router._assess_complexity(query)
        self.assertIsInstance(complexity, float)
        self.assertTrue(0 <= complexity <= 1)

        async def test_route():
            result = await self.router.route(query, [])
            self.assertIsInstance(result, dict)
            self.assertIn("model", result)
            self.assertIn("response_strategy", result)

        asyncio.run(test_route())

    def test_file_upload_and_rag(self):
        test_text = "This is a test document for the RAG system."
        self.rag_system.add_document(test_text, metadata={"filename": "test.txt"})

        result = self.rag_system.query("What is this document about?")
        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 0)

    def test_pinecone_operations(self):
        self.assertTrue(check_pinecone_health())

        # Test vector with random values
        test_vector = {
            "id": "test_vector",
            "values": [random.uniform(-1, 1) for _ in range(1536)],
            "metadata": {"test": "data"},
        }

        # Test upsert
        upsert_result = upsert_vectors([test_vector])
        self.assertIsNotNone(upsert_result)
        self.assertEqual(upsert_result["upserted_count"], 1)

        # Add a small delay to allow for indexing
        time.sleep(1)

        # Test query
        query_result = query_vectors(test_vector["values"], top_k=1)
        self.assertIsNotNone(query_result)
        self.assertTrue(len(query_result["matches"]) > 0)
        self.assertEqual(query_result["matches"][0]["id"], "test_vector")
        self.assertEqual(query_result["matches"][0]["metadata"], {"test": "data"})

        # Test delete
        delete_result = delete_vectors(["test_vector"])
        self.assertIsNotNone(delete_result)
        self.assertEqual(delete_result["deleted_count"], 1)

        # Test error handling for non-existent vector
        with self.assertRaises(Exception):
            query_vectors([0] * 1536, top_k=1)

        # Test upsert with invalid dimension
        invalid_vector = {
            "id": "invalid_vector",
            "values": [0.1] * 100,  # Invalid dimension
            "metadata": {"test": "invalid"},
        }
        with self.assertRaises(Exception):
            upsert_vectors([invalid_vector])

    def test_redis_cache(self):
        test_key = "test_key"
        test_value = {"data": "test_data"}

        RedisCache.set(test_key, test_value)
        retrieved_value = RedisCache.get(test_key)
        self.assertEqual(retrieved_value, test_value)

        # Test Docker Redis fallback
        RedisCache.redis_client = None
        RedisCache.set(test_key, test_value)
        retrieved_value = RedisCache.get(test_key)
        self.assertEqual(retrieved_value, test_value)

    def test_mongodb_operations(self):
        self.assertTrue(check_mongodb_health())

        test_document = {"test_field": "test_value"}
        insert_result = insert_document("test_collection", test_document)
        self.assertIsNotNone(insert_result)

        find_result = find_documents("test_collection", {"test_field": "test_value"})
        self.assertTrue(len(find_result) > 0)

        update_result = update_document(
            "test_collection",
            {"test_field": "test_value"},
            {"test_field": "updated_value"},
        )
        self.assertIsNotNone(update_result)

        delete_result = delete_document(
            "test_collection", {"test_field": "updated_value"}
        )
        self.assertIsNotNone(delete_result)

    def test_perplexity_search(self):
        query = "What is the population of New York City?"
        complexity = assess_complexity(query)
        self.assertIsInstance(complexity, float)
        self.assertTrue(0 <= complexity <= 1)

        search_result = perplexity_search(query, max_results=1)
        self.assertIsInstance(search_result, str)
        self.assertTrue(len(search_result) > 0)


if __name__ == "__main__":
    unittest.main()
