"""
AdvancedRouter: Dynamically selects the best model, parameters, and response
strategy for a given query or task.
"""

import logging
from typing import Dict, Any, List
from app.models import get_model_list, get_chat_model
from app.python.helpers.rate_limiter import RateLimiter
import re
from app.python.helpers.redis_cache import RedisCache
import uuid

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AdvancedRouter:
    def __init__(self, config: Dict[str, Any], agent):
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

        if task_type == "memory_retrieval":
            config = self._get_memory_config()
        elif task_type == "casual":
            config = self._get_casual_config()
        elif complexity < self.threshold / 2 and context_length < 1000:
            config = self._get_low_tier_config(task_type)
        elif complexity < self.threshold and context_length < 4000:
            config = self._get_mid_tier_config(task_type)
        elif complexity < self.threshold * 1.5 or context_length < 8000:
            config = self._get_high_tier_config(task_type)
        else:
            config = self._get_superior_tier_config(task_type)

        config["routing_explanation"] = (
            "Selected "
            + config["model"]
            + " based on complexity ("
            + "{:.2f}".format(complexity)
            + ") and context length ("
            + str(context_length)
            + " chars). "
            f"Threshold: {self.threshold}"
        )
        config["question_type"] = question_type
        config["response_strategy"] = self._get_response_strategy(
            question_type, task_type
        )
        config["task_type"] = task_type
        config["task_complexity"] = complexity

        config = self._adjust_params_based_on_history(config, conversation_history)

        logger.info(f"Final config: {config}")

        return config

    def _assess_complexity(self, query: str) -> float:
        word_count = len(query.split())
        sentence_count = len(re.findall(r"\w+[.!?]", query)) + 1
        avg_word_length = (
            sum(len(word) for word in query.split()) / word_count
            if word_count > 0
            else 0
        )

        complexity = (
            (word_count / 100) * 0.4
            + (sentence_count / 10) * 0.3
            + (avg_word_length / 10) * 0.3
        )
        return min(complexity, 1.0)

    def _calculate_context_length(
        self, conversation_history: List[Dict[str, str]]
    ) -> int:
        return sum(len(message["content"]) for message in conversation_history)

    def _identify_task_type(self, query: str) -> str:
        query_lower = query.lower()
        if any(
            word in query_lower for word in ["code", "program", "function", "debug"]
        ):
            return "coding"
        elif any(word in query_lower for word in ["analyze", "compare", "evaluate"]):
            return "analysis"
        elif any(word in query_lower for word in ["create", "generate", "write"]):
            return "creative"
        elif any(word in query_lower for word in ["hi", "hello", "hey", "how are you"]):
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
        elif any(word in query_lower for word in ["what", "who", "where", "when"]):
            return "factual"
        elif query_lower.startswith(("is", "are", "can", "do", "does")):
            return "yes_no"
        elif any(word in query_lower for word in ["compare", "contrast", "analyze"]):
            return "analysis"
        elif any(word in query_lower for word in ["hi", "hello", "hey", "how are you"]):
            return "casual"
        else:
            return "open_ended"

    def _get_response_strategy(self, question_type: str, task_type: str) -> str:
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
        if task_type in ["analysis", "creative"]:
            config["max_tokens"] = 768
        return config

    def _get_high_tier_config(self, task_type: str) -> Dict[str, Any]:
        config = {
            "model": self.model_tiers["high"],
            "max_tokens": 1024,
            "temperature": 0.9,
        }
        if task_type in ["coding", "analysis"]:
            config["temperature"] = 0.7
        return config

    def _get_superior_tier_config(self, task_type: str) -> Dict[str, Any]:
        config = {
            "model": self.model_tiers["superior"],
            "max_tokens": 8192,
            "temperature": 0.7,
        }
        if task_type in ["coding", "analysis"]:
            config["temperature"] = 0.5
        return config

    def _get_memory_config(self) -> Dict[str, Any]:
        return {
            "model": "memory_retrieval",
            "max_tokens": 512,
            "temperature": 0.3,
            "response_strategy": "memory_focused",
            "task_type": "memory_retrieval",
            "task_complexity": 0.6
        }

    def _adjust_params_based_on_history(
        self, config: Dict[str, Any], conversation_history: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        if len(conversation_history) > 5:
            config["temperature"] = min(config["temperature"] * 1.1, 1.0)

        if any(
            msg["content"].lower().startswith("please explain")
            for msg in conversation_history[-3:]
        ):
            config["max_tokens"] = min(int(config["max_tokens"] * 1.2), 8192)

        if len(conversation_history) >= 4 and all(
            len(msg["content"].split()) < 10 for msg in conversation_history[-4:]
        ):
            config["max_tokens"] = max(128, int(config["max_tokens"] * 0.8))

        return config

    async def process(
        self, query: str, model_name: str, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        try:
            conversation_history = params.get("conversation_history", [])
            config = await self.route(query, conversation_history)

            if not self.agent.chat_model:
                self.agent.initialize_models()

            # Check if the query is requesting to use tools
            if "use tool" in query.lower() or "access your tools" in query.lower():
                return await self.process_tool_request(query, config)

            # Check if the query is about current information
            if config["task_type"] == "current_info":
                return await self.process_online_knowledge_tool(query, params)

            # Create a messages list with the conversation history and the new query
            messages = [
                {
                    "role": "user" if msg["role"] == "user" else "assistant",
                    "content": msg["content"],
                }
                for msg in conversation_history
            ]
            messages.append({"role": "user", "content": query})

            # Get the chat model based on the config
            chat_model = get_chat_model(config["model"])

            # Use the invoke method with the correct arguments
            response = await chat_model.ainvoke(messages)

            return {
                "content": (
                    response.content if hasattr(response, "content") else str(response)
                ),
                "model_used": config["model"],
                "task_type": config["task_type"],
                "task_complexity": config["task_complexity"],
            }
        except Exception as e:
            logger.error(f"Error in AdvancedRouter process: {str(e)}", exc_info=True)
            raise

    async def process_tool_request(
        self, query: str, config: Dict[str, Any]
    ) -> Dict[str, Any]:
        available_tools = self.agent.tools.keys()
        tool_info = "Available tools:\n\n"

        for tool_name in available_tools:
            tool = self.agent.tools[tool_name]
            tool_info += f"- {tool_name}: {tool.description}\n"

        tool_info += (
            "\nTo use a tool, format your request as: [TOOL_NAME] Your request here"
        )

        return {
            "content": f"I understand you want to use tools. Here's what's available:\n\n{tool_info}",
            "model_used": config["model"],
            "task_type": "tool_use",
            "task_complexity": config["task_complexity"],
        }

    async def process_chat_model(
        self, query: str, model_name: str, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        try:
            await self.rate_limiter.acquire()
            try:
                messages = [{"role": "user", "content": query}]
                return await self.agent.chat_model.ainvoke(messages)
            finally:
                self.rate_limiter.release()
        except AttributeError as e:
            raise RuntimeError("Rate limiter is not properly initialized") from e

    async def process_knowledge_tool(
        self, query: str, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        knowledge_tool = self.agent.tools.get("knowledge_tool")
        if not knowledge_tool:
            return {"error": "Knowledge tool not found"}

        try:
            response = knowledge_tool.execute(question=query)
            return {"content": response.content, "tool_used": "knowledge_tool"}
        except Exception as e:
            return {"error": f"Error processing knowledge tool: {str(e)}"}

    async def process_memory_tool(
        self, query: str, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        memory_tool = self.agent.tools.get("memory_tool")
        if not memory_tool:
            return {"error": "Memory tool not found"}

        try:
            count = params.get("count", 3)
            threshold = params.get("threshold", 0.5)
            response = memory_tool.search(
                self.agent, query, count=count, threshold=threshold
            )
            return {"content": response, "tool_used": "memory_tool"}
        except Exception as e:
            return {"error": f"Error processing memory tool: {str(e)}"}

    async def process_online_knowledge_tool(
        self, query: str, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        online_knowledge_tool = self.agent.tools.get("online_knowledge_tool")
        if not online_knowledge_tool:
            logger.error("Online knowledge tool not found")
            return {"error": "Online knowledge tool not found"}

        try:
            logger.info(f"Processing query with online knowledge tool: {query}")
            response = online_knowledge_tool.run(query)
            logger.info(f"Online knowledge tool response: {response}")
            return {
                "content": (
                    response.message if hasattr(response, "message") else str(response)
                ),
                "model_used": "online_knowledge_tool",
                "task_type": "current_info",
                "task_complexity": 1.0,
            }
        except Exception as e:
            logger.error(
                f"Error processing online knowledge tool: {str(e)}", exc_info=True
            )
            return {"error": f"Error processing online knowledge tool: {str(e)}"}
