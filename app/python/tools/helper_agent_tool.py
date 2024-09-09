import time
from typing import List, Dict, Any
from app.python.helpers.tool import Tool, Response
from app.python.helpers.rate_limiter import RateLimiter


class HelperAgent:
    def __init__(
        self, model: str, rate_limit: int, max_input_tokens: int, max_output_tokens: int, window_seconds: int
    ):
        self.model = model
        self.rate_limiter = RateLimiter(rate_limit, max_input_tokens, max_output_tokens, window_seconds)

    def process(self, task: str) -> str:
        with self.rate_limiter:
            # Simulate API call to the helper agent
            time.sleep(1)  # Simulating processing time
            return f"Helper agent ({self.model}) processed: {task}"


helper_agents = [
    HelperAgent(
        "groq-base", 10, 1000, 500, 60
    ),  # 10 requests per minute, 1000 input tokens, 500 output tokens
    HelperAgent(
        "groq-small", 15, 1500, 750, 60
    ),  # 15 requests per minute, 1500 input tokens, 750 output tokens
    HelperAgent(
        "groq-large", 5, 2000, 1000, 60
    ),  # 5 requests per minute, 2000 input tokens, 1000 output tokens
]


class HelperAgentTool(Tool):
    name = "helper_agent_tool"

    def execute(self, **kwargs) -> Response:
        """
        Call multiple helper agents to process a task and return their responses.

        Args:
            task (str): The task to be processed by helper agents.

        Returns:
            Response: A Response object containing the results from helper agents.
        """
        task = kwargs.get("task", "")
        responses = {agent.model: agent.process(task) for agent in helper_agents}
        result = f"Processed task with {len(helper_agents)} helper agents"
        response_message = f"{result}\n\nHelper agent responses:\n"
        for model, response in responses.items():
            response_message += f"{model}: {response}\n"

        return Response(response_message, break_loop=False)
