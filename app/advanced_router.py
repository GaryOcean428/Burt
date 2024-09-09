"""
AdvancedRouter: Dynamically selects the best model, parameters, and response strategy for a given query or task.
"""

import logging
from typing import Dict, Any
from app.models import get_model_list
import random
from app.python.helpers.rate_limiter import RateLimiter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AdvancedRouter:
    def __init__(self, config: Dict[str, Any]):
        self.models = get_model_list()
        self.base_model = "llama-3.1-8b-instant"
        self.threshold = config.get("ROUTER_THRESHOLD", 0.7)
        self.rate_limiter = RateLimiter(
            max_calls=config.get("rate_limit_requests", 120),
            max_input_tokens=config.get("rate_limit_input_tokens", 200000),
            max_output_tokens=config.get("rate_limit_output_tokens", 200000),
            window_seconds=config.get("rate_limit_seconds", 60),
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
        lower_input = user_input.lower()
        if any(
            keyword in lower_input
            for keyword in ["code", "program", "function", "algorithm"]
        ):
            return "coding"
        elif any(
            keyword in lower_input
            for keyword in ["creative", "story", "imagine", "art"]
        ):
            return "creative"
        elif any(
            keyword in lower_input
            for keyword in ["analyze", "research", "compare", "evaluate"]
        ):
            return "analysis"
        elif any(
            keyword in lower_input
            for keyword in ["news", "current events", "latest", "today"]
        ):
            return "current_info"
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
        return (
            any(keyword in user_input.lower() for keyword in complex_keywords)
            or len(user_input.split()) > 50
        )

    def select_large_model(self, task_type: str) -> str:
        if task_type == "coding":
            return random.choice(["llama-3.1-70b-versatile", "gpt-4o"])
        elif task_type in ["creative", "analysis", "long_context"]:
            return "llama-3.1-70b-versatile"
        else:
            return "llama-3.1-70b-versatile"

    def select_small_model(self, task_type: str) -> str:
        if task_type in {"coding", "analysis"}:
            return random.choice(["llama-3.1-8b-instant", "mixtral-8x7b-32768"])
        elif task_type == "creative":
            return "llama-3.1-8b-instant"
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

        return params

    def get_model_capabilities(self, model: str) -> Dict[str, Any]:
        capabilities = {
            "max_tokens": 2048,  # Default value
            "strengths": [],
            "weaknesses": [],
        }

        if "llama3-groq" in model:
            capabilities |= {
                "max_tokens": 8192,
                "strengths": [
                    "Efficient for various tasks",
                    "Good balance of performance and resource usage",
                    "Optimized for tool use",
                ],
                "weaknesses": [
                    "Less general knowledge than some proprietary models"
                ],
            }
        elif "gpt-4" in model:
            capabilities.update(
                {
                    "max_tokens": 8192,
                    "strengths": [
                        "General knowledge",
                        "Complex reasoning",
                        "Code generation",
                    ],
                    "weaknesses": [
                        "Up-to-date information",
                        "Specialized domain knowledge",
                    ],
                }
            )
        elif "claude-3" in model:
            capabilities.update(
                {
                    "max_tokens": 100000,
                    "strengths": [
                        "Long context understanding",
                        "Analytical tasks",
                        "Detailed explanations",
                    ],
                    "weaknesses": ["Creative tasks", "Humor"],
                }
            )
        elif "llama-3" in model:
            capabilities.update(
                {
                    "max_tokens": 4096,
                    "strengths": [
                        "Open-source",
                        "Customizable",
                        "Efficient for certain tasks",
                    ],
                    "weaknesses": ["Less general knowledge than proprietary models"],
                }
            )
        elif "gemini" in model:
            capabilities.update(
                {
                    "max_tokens": 32768,
                    "strengths": [
                        "Multimodal capabilities",
                        "Up-to-date information",
                        "Efficient processing",
                    ],
                    "weaknesses": ["Less established than some other models"],
                }
            )
        elif "mistral" in model or "mixtral" in model:
            capabilities.update(
                {
                    "max_tokens": 8192,
                    "strengths": [
                        "Efficient for various tasks",
                        "Good balance of performance and resource usage",
                    ],
                    "weaknesses": [
                        "Less powerful than larger models for complex tasks"
                    ],
                }
            )

        return capabilities

    def estimate_task_complexity(self, user_input: str) -> float:
        complexity = len(user_input.split()) / 100
        complex_keywords = ["analyze", "compare", "evaluate", "synthesize", "optimize"]
        complexity += (
            sum(keyword in user_input.lower() for keyword in complex_keywords) * 0.2
        )
        return min(complexity, 1.0)

    def might_require_tool_use(self, user_input: str) -> bool:
        tool_use_keywords = [
            "use tool",
            "execute function",
            "call api",
            "perform action",
        ]
        return any(keyword in user_input.lower() for keyword in tool_use_keywords)

    def select_model_advanced(self, user_input: str) -> str:
        input_tokens = len(user_input.split())  # Simple estimation of input tokens
        call_record = self.rate_limiter.limit_call_and_input(input_tokens)
        if call_record:
            return self.base_model
        elif self.might_require_tool_use(user_input):
            return random.choice(
                [
                    "llama3-groq-70b-8192-tool-use-preview",
                    "llama3-groq-8b-8192-tool-use-preview",
                ]
            )
        else:
            return self.select_large_model(self.classify_task(user_input))

    def log_model_selection(
        self,
        selected_model: str,
        user_input: str,
        task_type: str,
        task_complexity: float,
    ):
        logger.info(f"Selected model: {selected_model}")
        logger.info(f"Task type: {task_type}")
        logger.info(f"Task complexity: {task_complexity:.2f}")
        logger.info(f"Input preview: {user_input[:100]}...")

    def process_query(self, user_input: str) -> str:
        try:
            logger.info(f"Processing query: {user_input[:100]}...")
            selected_model = self.select_model_advanced(user_input)
            logger.info(f"Selected model: {selected_model}")
            return selected_model
        except Exception as e:
            logger.error(f"Error in process_query: {str(e)}")
            raise

    def process_query_advanced(self, user_input: str) -> Dict[str, Any]:
        try:
            logger.info(f"Processing advanced query: {user_input[:100]}...")
            selected_model = self.process_query(user_input)
            task_type = self.classify_task(user_input)
            task_complexity = self.estimate_task_complexity(user_input)
            params = self.adjust_parameters(selected_model, task_type)

            self.log_model_selection(
                selected_model, user_input, task_type, task_complexity
            )

            return {
                "model": selected_model,
                "task_type": task_type,
                "task_complexity": task_complexity,
                "params": params,
            }
        except Exception as e:
            logger.error(f"Error in process_query_advanced: {str(e)}")
            raise

    def get_model_parameters(
        self, task_type: str, task_complexity: float
    ) -> Dict[str, Any]:
        params = {
            "temperature": 0.7,  # Default temperature
            "max_tokens": 2048,  # Default max tokens
        }

        if task_type == "coding":
            params["temperature"] = max(0.2, 0.7 - task_complexity * 0.5)
            params["max_tokens"] = min(4096, int(2048 + task_complexity * 2048))
        elif task_type == "creative":
            params["temperature"] = min(0.9, 0.7 + task_complexity * 0.2)
            params["max_tokens"] = min(4096, int(2048 + task_complexity * 2048))
        elif task_type == "analysis":
            params["temperature"] = max(0.3, 0.7 - task_complexity * 0.4)
            params["max_tokens"] = min(8192, int(2048 + task_complexity * 6144))
        elif task_type == "long_context":
            params["max_tokens"] = min(16384, int(4096 + task_complexity * 12288))
        else:  # general
            params["max_tokens"] = min(4096, int(2048 + task_complexity * 2048))

        return params
