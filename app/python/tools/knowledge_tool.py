from ...agent import Agent
from app.python.tools import online_knowledge_tool
from app.python.helpers import duckduckgo_search, perplexity_search
from app.python.tools import memory_tool
from app.python.helpers.tool import Tool, Response
from app.python.helpers import rate_limiter
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


class KnowledgeTool(Tool):
    name: str = "knowledge_tool"

    def __init__(self, agent: Agent):
        super().__init__(agent)
        self.rate_limiter = rate_limiter.RateLimiter(
            max_calls=self.agent.config.rate_limit_requests,
            max_input_tokens=self.agent.config.rate_limit_input_tokens,
            max_output_tokens=self.agent.config.rate_limit_output_tokens,
            window_seconds=self.agent.config.rate_limit_seconds,
        )

    def execute(self, **kwargs):
        question = kwargs.get("question", "")
        if not question:
            return Response("No question provided.", break_loop=False)

        with self.rate_limiter:
            memory_result = memory_tool.search(
                self.agent, question, count=3, threshold=0.5
            )
            if hasattr(online_knowledge_tool, "process_question"):
                online_result = online_knowledge_tool.process_question(
                    question, self.agent.config
                )
            else:
                online_result = "process_question is not available in online_knowledge_tool"

        combined_result = f"Memory: {memory_result}\n\nOnline: {online_result}"
        return Response(combined_result, break_loop=False)


class OnlineKnowledgeTool(Tool):
    def __init__(self, agent):
        super().__init__(agent)
        self.name = "online_knowledge_tool"
        self.description = "Searches for up-to-date information online using Perplexity Sonar models, with fallback to DuckDuckGo."

    def run(self, query: str) -> Response:
        config = self.agent.config
        result = self.process_question(query, config)
        return Response(message=result)

    def process_question(self, question: str, config: Dict[str, Any]) -> str:
        if perplexity_api_key := config.get("PERPLEXITY_API_KEY"):
            try:
                return perplexity_search.search_with_sonar(
                    question, perplexity_api_key
                )
            except Exception as e:
                logger.error(f"Perplexity Sonar search failed: {e}")

        # Fallback to DuckDuckGo
        return duckduckgo_search.search(question)


# Ensure the Tool class is available for import
Tool = KnowledgeTool
