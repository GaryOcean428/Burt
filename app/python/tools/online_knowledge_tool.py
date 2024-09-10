from app.python.helpers.tool import Tool, Response
from app.python.helpers import perplexity_search
from app.python.helpers import duckduckgo_search
from typing import Dict, Any
import logging
from app.agent_config import AgentConfig

logger = logging.getLogger(__name__)

class OnlineKnowledgeTool(Tool):
    def __init__(self, agent):
        super().__init__(agent)
        self.name = "online_knowledge_tool"
        self.description = "Searches for up-to-date information online using Perplexity Sonar models, with fallback to DuckDuckGo."

    def run(self, query: str) -> Response:
        config = self.agent.config
        result = self.process_question(query, config)
        return Response(message=result)

    def process_question(self, question: str, config: AgentConfig) -> str:
        perplexity_api_key = getattr(config, "PERPLEXITY_API_KEY", None)
        complexity = getattr(config, "task_complexity", 0.5)

        if perplexity_api_key:
            try:
                return perplexity_search.perplexity_search(question, api_key=perplexity_api_key, complexity=complexity)
            except Exception as e:
                logger.error(f"Perplexity Sonar search failed: {e}")

        # Fallback to DuckDuckGo
        return duckduckgo_search.search(question)

def create_tool(agent):
    return OnlineKnowledgeTool(agent)
