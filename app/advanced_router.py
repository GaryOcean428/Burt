"""
AdvancedRouter: Dynamically selects the best model, parameters, and response
strategy for a given query or task.
"""

import asyncio
import logging
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from app.python.helpers.rag_system import RAGSystem
from app.python.helpers.rate_limiter import RateLimiter
from app.python.helpers.model_utils import get_model_list, get_chat_model
from app.python.helpers.redis_cache import RedisCache

nltk.download("punkt")
nltk.download("stopwords")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


class PerformanceTracker:
    def __init__(self):
        self.performance_factors: Dict[str, float] = {
            "low": 1.0,
            "mid": 1.0,
            "high": 1.0,
        }

    def get_performance_factors(self) -> Dict[str, float]:
        return self.performance_factors

    def update_performance(self, tier: str, processing_time: float) -> None:
        # Implement logic to update performance factors based on processing time
        pass


class AdvancedRouter:
    def __init__(
        self,
        config: Dict[str, Any],
        agent: Any,
        rag_system: Optional[RAGSystem] = None,
    ):
        self.models = get_model_list()
        self.threshold = config.get("ROUTER_THRESHOLD", 0.7)
        self.rate_limiter = RateLimiter(
            max_calls=config.get("rate_limit_requests", 120),
            max_input_tokens=config.get("rate_limit_input_tokens", 200000),
            max_output_tokens=config.get("rate_limit_output_tokens", 200000),
            window_seconds=config.get("rate_limit_seconds", 60),
        )
        self.agent = agent
        self.model_tiers = {
            "low": "llama-3.1-8b-instant",
            "mid": "llama-3.1-70b-versatile",
            "high": "claude-3-opus-20240229",
            "superior": "claude-3-opus-20240229",
        }
        self.rag_system = rag_system or RAGSystem()
        self.model_performance: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {"total_time": 0.0, "count": 0, "avg_time": 0.0}
        )
        self.performance_tracker = PerformanceTracker()

    async def route(
        self, query: str, conversation_history: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        complexity = self._assess_complexity(query)
        context_length = self._calculate_context_length(conversation_history)
        task_type = self._identify_task_type(query)
        question_type = self._classify_question(query)

        logger.info(f"Query: {query}")
        logger.info(f"Complexity: {complexity}")
        logger.info(f"Context length: {context_length}")
        logger.info(f"Task type: {task_type}")
        logger.info(f"Question type: {question_type}")

        config = self._select_model_config(
            complexity, context_length, task_type
        )

        config["routing_explanation"] = (
            f"Selected {config['model']} based on complexity "
            f"({complexity:.2f}), context length ({context_length} chars), "
            f"and task type ({task_type}). Threshold: {self.threshold}"
        )
        config["question_type"] = question_type
        config["response_strategy"] = self._get_response_strategy(
            question_type, task_type
        )
        config["task_type"] = task_type
        config["task_complexity"] = complexity

        config = self._adjust_params_based_on_history(
            config, conversation_history
        )

        logger.info(f"Final config: {config}")

        return config

    def _assess_complexity(self, query: str) -> float:
        # Tokenize the query
        tokens = word_tokenize(query.lower())

        # Remove stopwords
        stop_words = set(stopwords.words("english"))
        tokens = [word for word in tokens if word not in stop_words]

        # Calculate lexical diversity
        lexical_diversity = len(set(tokens)) / len(tokens) if tokens else 0

        # Calculate average word length
        avg_word_length = (
            sum(len(word) for word in tokens) / len(tokens) if tokens else 0
        )

        # Count special characters and numbers
        special_chars = sum(
            1
            for char in query
            if not char.isalnum() and char not in [" ", ".", ",", "!", "?"]
        )
        numbers = sum(1 for char in query if char.isdigit())

        # Calculate complexity score
        complexity = (
            (len(tokens) / 100) * 0.3  # Length factor
            + lexical_diversity * 0.3  # Vocabulary richness
            + (avg_word_length / 10) * 0.2  # Word complexity
            + (special_chars / len(query)) * 0.1  # Special character density
            + (numbers / len(query)) * 0.1  # Number density
        )

        return min(complexity, 1.0)

    def _calculate_context_length(
        self, conversation_history: List[Dict[str, str]]
    ) -> int:
        return sum(len(message["content"]) for message in conversation_history)

    def _identify_task_type(self, query: str) -> str:
        query_lower = query.lower()
        if any(
            word in query_lower
            for word in ["code", "program", "function", "debug"]
        ):
            return "coding"
        elif any(
            word in query_lower for word in ["analyze", "compare", "evaluate"]
        ):
            return "analysis"
        elif any(
            word in query_lower for word in ["create", "generate", "write"]
        ):
            return "creative"
        elif any(
            word in query_lower
            for word in ["hi", "hello", "hey", "how are you"]
        ):
            return "casual"
        elif any(
            word in query_lower
            for word in ["news", "current events", "latest", "today"]
        ):
            return "current_info"
        else:
            return "general"

    def _classify_question(self, query: str) -> str:
        query_lower = query.lower()
        if any(word in query_lower for word in ["how", "why", "explain"]):
            return "problem_solving"
        elif any(
            word in query_lower for word in ["what", "who", "where", "when"]
        ):
            return "factual"
        elif query_lower.startswith(("is", "are", "can", "do", "does")):
            return "yes_no"
        elif any(
            word in query_lower for word in ["compare", "contrast", "analyze"]
        ):
            return "analysis"
        elif any(
            word in query_lower
            for word in ["hi", "hello", "hey", "how are you"]
        ):
            return "casual"
        else:
            return "open_ended"

    def _get_response_strategy(
        self, question_type: str, task_type: str
    ) -> str:
        if task_type == "casual" or question_type == "casual":
            return "casual_conversation"

        strategy_map = {
            "problem_solving": "chain_of_thought",
            "factual": "direct_answer",
            "yes_no": "boolean_with_explanation",
            "analysis": "comparative_analysis",
            "open_ended": "open_discussion",
        }
        return strategy_map.get(question_type, "default")

    def _select_model_config(self, complexity, context_length, task_type):
        performance_factor = self.performance_tracker.get_performance_factors()

        if (
            complexity >= self.threshold
            or context_length >= 4000
            or performance_factor.get("high", 1.0) <= 0.8
        ):
            return self._get_high_tier_config(task_type)
        elif (
            complexity < self.threshold
            and context_length < 4000
            and performance_factor.get("mid", 1.0) <= 1.2
        ):
            return self._get_mid_tier_config(task_type)
        else:
            return self._get_low_tier_config(task_type)

    def _get_performance_factor(self) -> Dict[str, float]:
        return {
            tier: (
                (
                    perf["avg_time"]
                    / min(
                        self.model_performance.values(),
                        key=lambda x: x["avg_time"],
                    )["avg_time"]
                )
                if perf["count"] > 0
                else 1.0
            )
            for tier, perf in self.model_performance.items()
        }

    def _get_casual_config(self) -> Dict[str, Any]:
        return {
            "model": self.model_tiers["low"],
            "max_tokens": 50,
            "temperature": 0.7,
            "response_strategy": "casual_conversation",
            "routing_explanation": "Simple greeting detected, using low-tier model for quick response.",
        }

    def _get_low_tier_config(self, task_type: str) -> Dict[str, Any]:
        config = {
            "model": self.model_tiers["low"],
            "max_tokens": 256,
            "temperature": 0.5,
        }
        if task_type == "casual":
            config["temperature"] = 0.7
        return config

    def _get_mid_tier_config(self, task_type: str) -> Dict[str, Any]:
        config = {
            "model": self.model_tiers["mid"],
            "max_tokens": 512,
            "temperature": 0.7,
        }
        if task_type in {"analysis", "creative"}:
            config["max_tokens"] = 768
        return config

    def _get_high_tier_config(self, task_type: str) -> Dict[str, Any]:
        config = {
            "model": self.model_tiers["high"],
            "max_tokens": 1024,
            "temperature": 0.9,
        }
        if task_type in {"coding", "analysis"}:
            config["temperature"] = 0.7
        return config

    def _get_superior_tier_config(self, task_type: str) -> Dict[str, Any]:
        config = {
            "model": self.model_tiers["superior"],
            "max_tokens": 8192,
            "temperature": 0.7,
        }
        if task_type in {"coding", "analysis"}:
            config["temperature"] = 0.5
        return config

    def _get_memory_config(self) -> Dict[str, Any]:
        return {
            "model": "memory_retrieval",
            "max_tokens": 512,
            "temperature": 0.3,
            "response_strategy": "memory_focused",
            "task_type": "memory_retrieval",
            "task_complexity": 0.6,
        }

    def _adjust_params_based_on_history(
        self,
        config: Dict[str, Any],
        conversation_history: List[Dict[str, str]],
    ) -> Dict[str, Any]:
        if len(conversation_history) > 5:
            config["temperature"] = min(config["temperature"] * 1.1, 1.0)

        if any(
            msg["content"].lower().startswith("please explain")
            for msg in conversation_history[-3:]
        ):
            config["max_tokens"] = min(int(config["max_tokens"] * 1.2), 8192)

        if len(conversation_history) >= 4 and all(
            len(msg["content"].split()) < 10
            for msg in conversation_history[-4:]
        ):
            config["max_tokens"] = max(128, int(config["max_tokens"] * 0.8))

        return config

    async def process(
        self, query: str, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        try:
            conversation_history = params.get("conversation_history", [])
            config = await self.route(query, conversation_history)

            if not self.agent.chat_model:
                self.agent.initialize_models()

            if (
                "use tool" in query.lower()
                or "access your tools" in query.lower()
            ):
                return await self.process_tool_request(query, config)

            if config["task_type"] == "current_info":
                return await self.process_online_knowledge_tool(query, config)

            messages = [
                {
                    "role": "user" if msg["role"] == "user" else "assistant",
                    "content": msg["content"],
                }
                for msg in conversation_history
            ]
            messages.append({"role": "user", "content": query})

            chat_model = get_chat_model(config["model"])

            start_time = time.time()
            response = await chat_model.ainvoke(messages)
            end_time = time.time()

            # Update model performance metrics
            self._update_model_performance(
                config["model"], end_time - start_time
            )

            return {
                "content": (
                    response.content
                    if hasattr(response, "content")
                    else str(response)
                ),
                "model_used": config["model"],
                "task_type": config["task_type"],
                "task_complexity": config["task_complexity"],
                "processing_time": end_time - start_time,
            }
        except Exception as e:
            logger.error(
                f"Error in AdvancedRouter process: {str(e)}", exc_info=True
            )
            raise

    def _update_model_performance(
        self, model: str, processing_time: float
    ) -> None:
        perf = self.model_performance[model]
        perf["total_time"] += processing_time
        perf["count"] += 1
        perf["avg_time"] = perf["total_time"] / perf["count"]

    async def process_tool_request(
        self, query: str, config: Dict[str, Any]
    ) -> Dict[str, Any]:
        available_tools = self.agent.tools.keys()
        tool_info = "Available tools:\n\n"

        for tool_name in available_tools:
            tool = self.agent.tools[tool_name]
            tool_info += f"- {tool_name}: {tool.description}\n"

        tool_info += "\nTo use a tool, format your request as: [TOOL_NAME] Your request here"

        return {
            "content": f"I understand you want to use tools. Here's what's available:\n\n{tool_info}",
            "model_used": config["model"],
            "task_type": "tool_use",
            "task_complexity": config["task_complexity"],
        }

    async def process_chat_model(
        self, query: str, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        try:
            await self.rate_limiter.acquire()
            try:
                messages = [{"role": "user", "content": query}]
                return await self.agent.chat_model.ainvoke(messages)
            finally:
                self.rate_limiter.release()
        except AttributeError as e:
            logger.error(f"Rate limiter not properly initialized: {str(e)}")
            raise RuntimeError(
                "Rate limiter is not properly initialized"
            ) from e

    async def process_knowledge_tool(
        self, query: str, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        knowledge_tool = self.agent.tools.get("knowledge_tool")
        if not knowledge_tool:
            logger.error("Knowledge tool not found")
            return {"error": "Knowledge tool not found"}

        try:
            response = knowledge_tool.execute(question=query)
            return {"content": response.content, "tool_used": "knowledge_tool"}
        except Exception as e:
            logger.error(
                f"Error processing knowledge tool: {str(e)}", exc_info=True
            )
            return {"error": f"Error processing knowledge tool: {str(e)}"}

    async def process_memory_tool(
        self, query: str, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        memory_tool = self.agent.tools.get("memory_tool")
        if not memory_tool:
            logger.error("Memory tool not found")
            return {"error": "Memory tool not found"}

        try:
            count = params.get("count", 3)
            threshold = params.get("threshold", 0.5)
            response = memory_tool.search(
                self.agent, query, count=count, threshold=threshold
            )
            return {"content": response, "tool_used": "memory_tool"}
        except Exception as e:
            logger.error(
                f"Error processing memory tool: {str(e)}", exc_info=True
            )
            return {"error": f"Error processing memory tool: {str(e)}"}

    async def process_online_knowledge_tool(
        self, query: str, config: Dict[str, Any]
    ) -> Dict[str, Any]:
        online_knowledge_tool = self.agent.tools.get("online_knowledge_tool")
        if not online_knowledge_tool:
            logger.error("Online knowledge tool not found")
            return {"error": "Online knowledge tool not found"}

        try:
            logger.info(
                f"Processing query with online knowledge tool: {query}"
            )

            hybrid_response = self.rag_system.hybrid_query(query)

            logger.info(f"Hybrid query response: {hybrid_response}")

            return {
                "content": hybrid_response,
                "model_used": "hybrid_rag_system",
                "task_type": "current_info",
                "task_complexity": config["task_complexity"],
            }
        except Exception as e:
            logger.error(
                f"Error processing online knowledge tool: {str(e)}",
                exc_info=True,
            )
            return {
                "error": f"Error processing online knowledge tool: {str(e)}"
            }

    def save_to_redis_cache(self, key: str, value: Any) -> None:
        try:
            RedisCache.set(key, value)
        except Exception as e:
            logger.error(
                f"Error saving to Redis cache: {str(e)}", exc_info=True
            )

    def get_from_redis_cache(self, key: str) -> Any:
        try:
            return RedisCache.get(key)
        except Exception as e:
            logger.error(
                f"Error retrieving from Redis cache: {str(e)}", exc_info=True
            )
            return None
