import unittest
from unittest.mock import patch
from app.agent import Agent, AgentConfig
from app.config import load_config
import os
from app.python.helpers.vdb import Document
from app.vector_db import VectorDB


class TestAgent(unittest.TestCase):
    def setUp(self):
        self.config = load_config()
        self.agent = Agent(1, self.config)

    def test_process_simple_input(self):
        result = self.agent.process("Hello, how are you?")
        self.assertIsInstance(result, str)
        self.assertNotEqual(result, "")

    def test_process_complex_input(self):
        result = self.agent.process(
            "Explain the theory of relativity and its implications in modern physics."
        )
        self.assertIsInstance(result, str)
        self.assertIn("relativity", result.lower())

    @patch("app.agent.Agent.use_tool")
    def test_tool_usage(self, mock_use_tool):
        mock_use_tool.return_value = "Tool executed successfully"
        result = self.agent.process(
            "Use the knowledge tool to find information about Python."
        )
        self.assertIn("Tool executed successfully", result)

    def test_intervention_handling(self):
        self.agent.set_intervention_status(True, "Test intervention")
        result = self.agent.process("Continue with the task")
        self.assertIn("Intervention received", result)
        self.assertFalse(self.agent.get_intervention_status())

    def test_dynamic_tool_loading(self):
        tools = self.agent.get_tools()
        self.assertIsInstance(tools, dict)
        self.assertGreater(len(tools), 0)

    @patch("app.agent.Agent.use_tool")
    def test_rate_limiting(self, mock_use_tool):
        mock_use_tool.return_value = "Tool executed"
        self.agent.config.rate_limit_requests = 2
        self.agent.process("Test input 1")
        self.agent.process("Test input 2")
        result = self.agent.process("Test input 3")
        self.assertIn("rate limit", result.lower())

    def test_save_and_load_state(self):
        self.agent.set_data_item("test_key", "test_value")
        self.agent.save_state("test_state.json")

        new_agent = Agent(2, self.config)
        new_agent.load_state("test_state.json")

        self.assertEqual(new_agent.get_data_item("test_key"), "test_value")

        os.remove("test_state.json")

    def test_memory_management(self):
        initial_history_length = len(self.agent.get_history())
        messages = [
            f"Test input {i}" for i in range(self.agent.config.rate_limit_msgs + 5)
        ]
        for message in messages:
            self.agent.process(message)
        final_history_length = len(self.agent.get_history())
        self.assertLessEqual(final_history_length, self.agent.config.msgs_keep_max)

    @patch("app.agent.Agent.use_tool")
    def test_planner_and_executor(self, mock_use_tool):
        mock_use_tool.return_value = "Planned step executed"
        result = self.agent.process(
            "Create a plan to build a website and execute the first step"
        )
        self.assertIn("Planned step executed", result)

    def test_error_handling(self):
        with patch("app.agent.Agent.use_tool", side_effect=Exception("Test error")):
            result = self.agent.process("Trigger an error")
            self.assertIn("An error occurred", result)

    def test_vector_operations(self):
        config = AgentConfig(
            chat_model="test_model", embeddings_model="text-embedding-ada-002"
        )
        agent = Agent(1, config)
        agent.initialize_models()

        # Test adding a document
        doc = Document("1", "This is a test document")
        agent.vector_db.add_documents([doc])

        # Test searching
        agent.vector_db = (
            VectorDB()
        )  # Replace VectorDB with the actual class name of the vector database
        results = agent.vector_db.search("test document")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].content, "This is a test document")

        # Test deleting a document
        agent.vector_db.delete_document("1")
        results = agent.vector_db.search("test document")
        self.assertEqual(len(results), 0)


if __name__ == "__main__":
    unittest.main()
