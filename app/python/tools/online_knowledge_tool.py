from app.python.helpers.tool import Tool, Response
from app.python.helpers import perplexity_search
from app.python.helpers import duckduckgo_search
from app.python.helpers.rag_system import RAGSystem
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class OnlineKnowledgeTool(Tool):
    def __init__(self, agent):
        super().__init__(agent)
        self.name = "online_knowledge_tool"
        self.description = "Searches for up-to-date information online using Perplexity Sonar models, with fallback to DuckDuckGo and RAG system."

    def run(self, query: str) -> Response:
        config = self.agent.config
        result = self.process_question(query, config)
        return Response(message=result)

    def process_question(self, question: str, config: Dict[str, Any]) -> str:
        # First, try to retrieve information from RAG system
        rag_system = RAGSystem()
        rag_response = rag_system.query(question)

        if rag_response:
            return f"From memory: {rag_response}\n\nAdditional online search:"

        # Then, try Perplexity search
        if perplexity_api_key := config.get("perplexity_api_key"):
            try:
                return perplexity_search.perplexity_search(
                    question, api_key=perplexity_api_key
                )
            except Exception as e:
                logger.error(f"Perplexity search failed: {e}")

        # Fallback to DuckDuckGo
        return duckduckgo_search.search(question)


def create_tool(agent):
    return OnlineKnowledgeTool(agent)
