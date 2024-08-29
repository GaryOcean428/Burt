import time
from typing import List, Dict, Any
from python.helpers.tool import tool, Response
from python.helpers.rate_limiter import RateLimiter


class HelperAgent:
    def __init__(self, model: str, rate_limit: int):
        self.model = model
        self.rate_limiter = RateLimiter(rate_limit)

    def process(self, task: str) -> str:
        with self.rate_limiter:
            # Simulate API call to the helper agent
            time.sleep(1)  # Simulating processing time
            return f"Helper agent ({self.model}) processed: {task}"


helper_agents = [
    HelperAgent("groq-base", 10),  # 10 requests per minute
    HelperAgent("groq-small", 15),  # 15 requests per minute
    HelperAgent("groq-large", 5),  # 5 requests per minute
]


@tool
def call_helper_agents(tool) -> Response:
    """
    Call multiple helper agents to process a task and return their responses.

    Args:
        task (str): The task to be processed by helper agents.

    Returns:
        Response: A Response object containing the results from helper agents.
    """
    task = tool.args.get("task", "")
    responses = {agent.model: agent.process(task) for agent in helper_agents}
    result = f"Processed task with {len(helper_agents)} helper agents"
    response_message = f"{result}\n\nHelper agent responses:\n"
    for model, response in responses.items():
        response_message += f"{model}: {response}\n"

    return Response(response_message, break_loop=False)
