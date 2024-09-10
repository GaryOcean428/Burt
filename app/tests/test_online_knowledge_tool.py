import unittest
from unittest.mock import MagicMock, patch
from app.python.tools.online_knowledge_tool import OnlineKnowledgeTool
from app.python.helpers import perplexity_search, duckduckgo_search


class TestOnlineKnowledgeTool(unittest.TestCase):
    def setUp(self):
        self.agent_mock = MagicMock()
        self.agent_mock.config = {"PERPLEXITY_API_KEY": "test_key"}
        self.tool = OnlineKnowledgeTool(self.agent_mock)

    @patch("app.python.helpers.perplexity_search.perplexity_search")
    def test_perplexity_search(self, mock_search):
        mock_search.return_value = "Perplexity search result"
        result = self.tool.process_question(
            "What is the capital of France?", self.agent_mock.config
        )
        self.assertEqual(result, "Perplexity search result")
        mock_search.assert_called_once_with("What is the capital of France?")

    @patch("app.python.helpers.perplexity_search.perplexity_search")
    @patch("app.python.helpers.duckduckgo_search.search")
    def test_fallback_to_duckduckgo(self, mock_ddg_search, mock_perplexity_search):
        mock_perplexity_search.side_effect = Exception("Perplexity API error")
        mock_ddg_search.return_value = "DuckDuckGo search result"
        result = self.tool.process_question(
            "What is the capital of France?", self.agent_mock.config
        )
        self.assertEqual(result, "DuckDuckGo search result")
        mock_perplexity_search.assert_called_once()
        mock_ddg_search.assert_called_once_with("What is the capital of France?")


if __name__ == "__main__":
    unittest.main()
