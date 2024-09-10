from app.agent import Agent
from app.python.tools import online_knowledge_tool
from app.python.helpers import duckduckgo_search
from app.python.tools import memory_tool
import concurrent.futures
from app.python.helpers.tool import Tool, Response
from app.python.helpers import files
from app.python.helpers.print_style import PrintStyle
from app.python.helpers import rate_limiter
import logging

logger = logging.getLogger(__name__)


class KnowledgeTool(Tool):
    name = "knowledge_tool"  # type: str

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
            if hasattr(online_knowledge_tool, 'process_question'):
                online_result = online_knowledge_tool.process_question(
                    question, self.agent.config
                )
            else:
                online_result = (
                    "process_question is not available "
                    "in online_knowledge_tool"
                )

        combined_result = f"Memory: {memory_result}\n\nOnline: {online_result}"
        return Response(combined_result, break_loop=False)


# Ensure the Tool class is available for import
Tool = KnowledgeTool
