"""
AdvancedRouter: Dynamically selects the best model, parameters, and response
strategy for a given query or task.
"""

import logging
from typing import Dict, Any, Union, List
from app.models import get_model_list
import random
from app.python.helpers.rate_limiter import RateLimiter
import asyncio
from typing import List, Dict, Any
from app.python.tools import knowledge_tool, memory_tool, online_knowledge_tool
from app.models import get_chat_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AdvancedRouter:
    def __init__(self, config: Dict[str, Any]):
        self.models = get_model_list()
        self.base_model = "llama-3.1-8b-instant"  # Default base model
        self.threshold = config.get("ROUTER_THRESHOLD", 0.7)
        self.rate_limiter = RateLimiter(
            max_calls=config.get("rate_limit_requests", 120),
            max_input_tokens=config.get("rate_limit_input_tokens", 200000),
            max_output_tokens=config.get("rate_limit_output_tokens", 200000),
            window_seconds=config.get("rate_limit_seconds", 60),
        )
        self.allowed_models = {
            "gpt-4o",
            "gpt-4o-mini",
            "claude-3-5-sonnet-20240620",
            "claude-3-opus-20240229",
            "llama-3.1-8b-instant",
            "llama-3.1-70b-versatile",
            "llama-3.1-sonar-small-128k-online",
            "llama-3.1-sonar-large-128k-online",
            "llama-3.1-sonar-huge-128k-online",
            "llama3-groq-70b-8192-tool-use-preview",
            "llama3-groq-8b-8192-tool-use-preview",
        }
        self.keyword_categories = {
            "coding": {"code", "program", "function", "algorithm"},
            "creative": {"creative", "story", "imagine", "art"},
            "analysis": {"analyze", "research", "compare", "evaluate"},
            "current_info": {"news", "current events", "latest", "today"},
            "complex": {
                "analyze",
                "explain",
                "compare",
                "synthesize",
                "evaluate",
                "design",
                "optimize",
                "predict",
                "simulate",
            },
            "tool_use": {
                "use tool",
                "execute function",
                "call api",
                "perform action"
            },
        }

    def select_model(self, user_input: str) -> str:
        input_tokens = (
            len(user_input.split())  # Simple estimation of input tokens
        )
        call_record = self.rate_limiter.limit_call_and_input(input_tokens)

        if call_record is None:
            return self.base_model

        task_type = self.classify_task(user_input)
        if task_type == "current_info":
            return (
                "llama-3.1-sonar-small-128k-online"  # or another suitable
                " online model"
            )
        elif self.is_complex_task(user_input):
            return self.select_large_model(task_type)
        else:
            return self.select_small_model(task_type)

    def classify_task(self, user_input: str) -> str:
        lower_input = user_input.lower()
        words = set(lower_input.split())
        return next(
            (
                category
                for category, keywords in self.keyword_categories.items()
                if words & keywords
            ),
            "general",
        )

    def is_complex_task(self, user_input: str) -> bool:
        words = set(user_input.lower().split())
        return (bool(words & self.keyword_categories["complex"]) or
                len(words) > 50)

    def select_large_model(self, task_type: str) -> str:
        if task_type == "coding":
            return random.choice(["llama-3.1-70b-versatile", "gpt-4o"])
        else:
            return random.choice([
                "llama-3.1-70b-versatile",
                "claude-3-opus-20240229"
            ])

    def select_small_model(self, task_type: str) -> str:
        if task_type in {"coding", "analysis"}:
            return random.choice(["llama-3.1-8b-instant", "gpt-4o-mini"])
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

        if model not in self.allowed_models:
            logger.warning(f"Model {model} is not allowed.")
            # Return default capabilities if model is not allowed
            return capabilities

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
        elif "gpt-4o" in model:  # Updated model name
            capabilities |= {
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
        elif "claude-3" in model:
            capabilities |= {
                "max_tokens": 100000,
                "strengths": [
                    "Long context understanding",
                    "Analytical tasks",
                    "Detailed explanations",
                ],
                "weaknesses": ["Creative tasks", "Humor"],
            }
        elif "llama-3.1" in model:  # Ensure we are using the correct version
            capabilities |= {
                "max_tokens": 4096,
                "strengths": [
                    "Open-source",
                    "Customizable",
                    "Efficient for certain tasks",
                ],
                "weaknesses": [
                    "Less general knowledge than proprietary models"
                ],
            }
        elif "gemini" in model:
            capabilities |= {
                "max_tokens": 32768,
                "strengths": [
                    "Multimodal capabilities",
                    "Up-to-date information",
                    "Efficient processing",
                ],
                "weaknesses": ["Less established than some other models"],
            }
        elif "mistral" in model or "mixtral" in model:
            capabilities |= {
                "max_tokens": 8192,
                "strengths": [
                    "Efficient for various tasks",
                    "Good balance of performance and resource usage",
                ],
                "weaknesses": [
                    "Less powerful than larger models for complex tasks"
                ],
            }

        return capabilities

    def estimate_task_complexity(self, user_input: str) -> float:
        complexity = len(user_input.split()) / 100
        complex_keywords = set(self.keyword_categories["complex"])
        complexity += sum(
            keyword in user_input.lower() for keyword in complex_keywords
        ) * 0.2

        # Check for potential agentic tasks
        agentic_keywords = {"plan", "execute", "analyze", "solve", "implement"}
        if any(keyword in user_input.lower() for keyword in agentic_keywords):
            complexity += 0.3

        return min(complexity, 1.0)

    def might_require_tool_use(self, user_input: str) -> bool:
        return any(
            keyword in user_input.lower()
            for keyword in self.keyword_categories["tool_use"]
        )

    def select_model_advanced(self, user_input: str) -> str:
        # Simple estimation of input tokens
        input_tokens = len(user_input.split())
        if call_record := self.rate_limiter.limit_call_and_input(input_tokens):
            return self.base_model
        elif self.might_require_tool_use(user_input):
            return random.choice([
                "llama3-groq-70b-8192-tool-use-preview",
                "llama3-groq-8b-8192-tool-use-preview",
            ])
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

    async def run_llm(self, model: str, user_input: str) -> str:
        """Run a single LLM call with a reference model."""
        chat_model = get_chat_model(model, temperature=0.7)
        response = await chat_model.ainvoke([{"role": "user", "content": user_input}])
        return response.content

    async def moa_process(self, user_input: str) -> Dict[str, Any]:
        """Process the query using Mixture of Agents approach."""
        reference_models = [
            "llama-3.1-8b-instant",
            "llama-3.1-70b-versatile",
            "claude-3-5-sonnet-20240620",
            "gpt-4o-mini",
        ]
        aggregator_model = "llama-3.1-70b-versatile"

        results = await asyncio.gather(
            *[self.run_llm(model, user_input) for model in reference_models]
        )

        aggregator_prompt = f"""You have been provided with responses from various
        models to the query: "{user_input}"
        Your task is to synthesize these responses into a single, high-quality
        response. Critically evaluate the information, recognizing that some of
        it may be biased or incorrect. Offer a refined, accurate, and
        comprehensive reply.

        Responses from models: {', '.join(results)}"""

        final_response = await self.run_llm(aggregator_model, aggregator_prompt)
        return {"response": final_response, "model_used": "MoA"}

    async def process_query_advanced(self, user_input: str) -> Dict[str, Any]:
        try:
            logger.info(f"Processing advanced query: {user_input[:100]}...")
            task_type = self.classify_task(user_input)
            task_complexity = self.estimate_task_complexity(user_input)

            if task_complexity > 0.6:  # Use MoA for complex queries
                result = await self.moa_process(user_input)
                selected_model = "MoA"
            else:
                selected_model = self.select_model(user_input)
                result = {"response": "", "model_used": selected_model}

            params = self.get_model_parameters(task_type, task_complexity)

            # Incorporate knowledge tool for all queries
            knowledge_result = knowledge_tool.KnowledgeTool(self.agent).execute(
                question=user_input
            )

            if isinstance(knowledge_result.content, str):
                memory_part, online_part = knowledge_result.content.split(
                    "\n\nOnline: "
                )
                memory_info = memory_part.replace("Memory: ", "").strip()
                online_info = online_part.strip()
            else:
                memory_info = "No relevant memory found."
                online_info = "No online information retrieved."

            # Combine MoA/model response with knowledge tool results
            combined_response = f"""Model response: {result['response']}

Relevant memory: {memory_info}

Online information: {online_info}

Based on the model response, relevant memory, and current online information, here's a comprehensive answer:
[Insert final synthesized response here]"""

            self.log_model_selection(
                selected_model, user_input, task_type, task_complexity
            )

            return {
                "model": selected_model,
                "task_type": task_type,
                "task_complexity": task_complexity,
                "params": params,
                "response": combined_response,
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
