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
        self.description = "Searches for up-to-date information using a hybrid approach with RAG system, Perplexity Sonar models, and DuckDuckGo as fallback."
        self.rag_system = RAGSystem()

    def run(self, query: str) -> Response:
        config = self.agent.config
        result = self.process_question(query, config)
        return Response(message=result)

    def process_question(self, question: str, config: Dict[str, Any]) -> str:
        try:
            # Use the hybrid query approach from RAG system
            hybrid_response = self.rag_system.hybrid_query(question)
            return f"Hybrid search result:\n{hybrid_response}"
        except Exception as e:
            logger.error(f"Hybrid query failed: {e}")

            # Fallback to DuckDuckGo if hybrid query fails
            duckduckgo_response = self.try_duckduckgo_search(question)
            if duckduckgo_response:
                return f"DuckDuckGo search result:\n{duckduckgo_response}"

            return "I'm sorry, but I couldn't retrieve any information at the moment. Please try again later."

    def try_duckduckgo_search(self, question: str) -> str:
        try:
            return duckduckgo_search.search(question, timeout=30)
        except Exception as e:
            logger.error(f"DuckDuckGo search failed: {e}")
            return ""


def create_tool(agent):
    return OnlineKnowledgeTool(agent)
