from app.agent import Agent
from . import online_knowledge_tool
from python.helpers import perplexity_search
from python.helpers import duckduckgo_search
from . import memory_tool
import concurrent.futures
from python.helpers.tool import Tool, Response
from python.helpers import files
from python.helpers.print_style import PrintStyle
from python.helpers import rate_limiter


class KnowledgeTool(Tool):
    name = "knowledge_tool"

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
            memory_result = memory_tool.search(self.agent, question)
            online_result = online_knowledge_tool.process_question(
                question, self.agent.config
            )

        combined_result = f"Memory: {memory_result}\n\nOnline: {online_result}"
        return Response(combined_result, break_loop=False)


# Remove the duplicate Agent class definition
