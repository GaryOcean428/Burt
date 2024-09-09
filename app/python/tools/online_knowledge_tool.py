from app.python.helpers.tool import Tool, Response
from python.helpers import perplexity_search
from python.helpers import duckduckgo_search
from typing import Dict, Any


class OnlineKnowledgeTool(Tool):
    def __init__(self, agent):
        super().__init__(agent)
        self.name = "online_knowledge_tool"
        self.description = "Searches for up-to-date information online using Perplexity Sonar models, with fallback to DuckDuckGo."

    def run(self, query: str) -> Response:
        config = self.agent.config
        result = self.process_question(query, config)
        return Response(content=result)

    def process_question(self, question: str, config: Dict[str, Any]) -> str:
        if perplexity_api_key := config.get("PERPLEXITY_API_KEY"):
            try:
                return perplexity_search.search_with_sonar(question, perplexity_api_key)
            except Exception as e:
                print(f"Perplexity Sonar search failed: {e}")

        # Fallback to DuckDuckGo
        return duckduckgo_search.search(question)


def create_tool(agent):
    return OnlineKnowledgeTool(agent)
