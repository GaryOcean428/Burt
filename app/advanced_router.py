"""
AdvancedRouter: Dynamically selects the best model, parameters, and response strategy for a given query or task.
"""

from typing import Dict, Any, List
import re
import logging
from app.models import get_model_list
from app.config import ROUTER_THRESHOLD
import random
from app.python.helpers.rate_limiter import RateLimiter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AdvancedRouter:
    def __init__(self):
        self.models = get_model_list()
        self.base_model = "llama-3.1-8b"
        self.threshold = ROUTER_THRESHOLD
        self.rate_limiter = RateLimiter(
            max_calls=120,  # Increased from 60
            max_input_tokens=200000,  # Increased from 100000
            max_output_tokens=200000,  # Increased from 100000
            window_seconds=60,
        )

    def select_model(self, user_input: str) -> str:
        input_tokens = len(user_input.split())  # Simple estimation of input tokens
        call_record = self.rate_limiter.limit_call_and_input(input_tokens)

        if call_record is None:
            return self.base_model

        task_type = self.classify_task(user_input)
        if self.is_complex_task(user_input):
            return self.select_large_model(task_type)
        else:
            return self.select_small_model(task_type)

    def classify_task(self, user_input: str) -> str:
        if any(
            keyword in user_input.lower()
            for keyword in ["code", "program", "function", "algorithm"]
        ):
            return "coding"
        elif any(
            keyword in user_input.lower()
            for keyword in ["creative", "story", "imagine", "art"]
        ):
            return "creative"
        elif any(
            keyword in user_input.lower()
            for keyword in ["analyze", "research", "compare", "evaluate"]
        ):
            return "analysis"
        elif len(user_input.split()) > 100:
            return "long_context"
        else:
            return "general"

    def is_complex_task(self, user_input: str) -> bool:
        complex_keywords = [
            "analyze",
            "explain",
            "compare",
            "synthesize",
            "evaluate",
            "design",
            "optimize",
            "predict",
            "simulate",
        ]
        if any(keyword in user_input.lower() for keyword in complex_keywords):
            return True
        if len(user_input.split()) > 50:  # Increased threshold for longer inputs
            return True
        return False

    def select_large_model(self, task_type: str) -> str:
        if task_type == "coding":
            return random.choice(["gpt-4o", "claude-3.5-sonnet"])
        elif task_type == "creative":
            return random.choice(["gpt-4o", "claude-3.5-sonnet", "llama-3.1-70b"])
        elif task_type == "analysis":
            return random.choice(["gpt-4o", "claude-3.5-sonnet", "llama-3.1-405b"])
        elif task_type == "long_context":
            return random.choice(
                [
                    "llama-3.1-sonar-large-128k-online",
                    "llama-3.1-sonar-huge-128k-online",
                ]
            )
        else:
            return random.choice(
                ["gpt-4o", "claude-3.5-sonnet", "llama-3.1-405b", "gemini-1.5-pro"]
            )

    def select_small_model(self, task_type: str) -> str:
        if task_type == "coding":
            return random.choice(["gpt-4o-mini", "mixtral-8x7b"])
        elif task_type == "creative":
            return random.choice(["llama-3.1-8b", "mistral-large-2"])
        elif task_type == "analysis":
            return random.choice(["gpt-4o-mini", "mixtral-8x7b"])
        else:
            return self.base_model

    def adjust_parameters(self, model: str, task: str) -> Dict[str, Any]:
        params = {}
        if "code" in task.lower():
            params["temperature"] = 0.2
        elif "creative" in task.lower():
            params["temperature"] = 0.8
        else:
            params["temperature"] = 0.5

        if "llama" in model or "groq" in model:
            params["max_new_tokens"] = 2048
        elif "gpt" in model:
            params["max_tokens"] = 4096
        elif "claude" in model:
            params["max_tokens"] = 8192
        elif "gemini" in model:
            params["max_tokens"] = 16384
        elif "mistral" in model or "mixtral" in model:
            params["max_tokens"] = 4096

        return params


advanced_router = AdvancedRouter()
