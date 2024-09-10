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
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AdvancedRouter:
    def __init__(self, config: Dict[str, Any], agent):
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
            "tool_use": {"use tool", "execute function", "call api", "perform action"},
        }
        self.agent = agent

    async def route(self, query: str, conversation_history: List[Dict[str, str]]) -> Dict[str, Any]:
        complexity = self._assess_complexity(query)
        context_length = self._calculate_context_length(conversation_history)
        task_type = self._identify_task_type(query)
        question_type = self._classify_question(query)

        if complexity < self.threshold / 2 and context_length < 1000:
            config = self._get_low_tier_config(task_type)
        elif complexity < self.threshold and context_length < 4000:
            config = self._get_mid_tier_config(task_type)
        else:
            config = self._get_high_tier_config(task_type)

        config['routing_explanation'] = f"Selected {config['model']} based on complexity ({complexity:.2f}) and context length ({context_length} chars). Threshold: {self.threshold}"
        config['question_type'] = question_type
        config['response_strategy'] = self._get_response_strategy(question_type)

        return config

    def _assess_complexity(self, query: str) -> float:
        word_count = len(query.split())
        sentence_count = len(re.findall(r'\w+[.!?]', query)) + 1
        avg_word_length = sum(len(word) for word in query.split()) / word_count if word_count > 0 else 0

        complexity = (word_count / 100) * 0.4 + (sentence_count / 10) * 0.3 + (avg_word_length / 10) * 0.3
        return min(complexity, 1.0)

    def _calculate_context_length(self, conversation_history: List[Dict[str, str]]) -> int:
        return sum(len(message['content']) for message in conversation_history)

    def _identify_task_type(self, query: str) -> str:
        query_lower = query.lower()
        for category, keywords in self.keyword_categories.items():
            if any(keyword in query_lower for keyword in keywords):
                return category
        return "general"

    def _classify_question(self, query: str) -> str:
        query_lower = query.lower()
        if any(word in query_lower for word in ['how', 'why', 'explain']):
            return "problem_solving"
        elif any(word in query_lower for word in ['what', 'who', 'where', 'when']):
            return "factual"
        elif query_lower.startswith(('is', 'are', 'can', 'do', 'does')):
            return "yes_no"
        elif any(word in query_lower for word in ['compare', 'contrast', 'analyze']):
            return "analysis"
        else:
            return "open_ended"

    def _get_response_strategy(self, question_type: str) -> str:
        strategy_map = {
            "problem_solving": "irac",
            "factual": "direct_answer",
            "yes_no": "boolean_with_explanation",
            "analysis": "comparative_analysis",
            "open_ended": "open_discussion"
        }
        return strategy_map.get(question_type, "default")

    def _get_low_tier_config(self, task_type: str) -> Dict[str, Any]:
        return {
            'model': "llama-3.1-8b-instant",
            'max_tokens': 256,
            'temperature': 0.5,
        }

    def _get_mid_tier_config(self, task_type: str) -> Dict[str, Any]:
        return {
            'model': "llama-3.1-70b-versatile",
            'max_tokens': 512,
            'temperature': 0.7,
        }

    def _get_high_tier_config(self, task_type: str) -> Dict[str, Any]:
        return {
            'model': "claude-3-opus-20240229",
            'max_tokens': 1024,
            'temperature': 0.9,
        }

    async def process(self, query: str, model_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        if model_name not in self.allowed_models:
            return {"error": f"Model '{model_name}' not allowed."}

        if not self.agent.chat_model:
            self.agent.initialize_models()

        if model_name == "knowledge_tool":
            return await self.process_knowledge_tool(query, params)
        elif model_name == "memory_tool":
            return await self.process_memory_tool(query, params)
        elif model_name == "online_knowledge_tool":
            return await self.process_online_knowledge_tool(query, params)
        else:
            return await self.process_chat_model(query, model_name, params)

    async def process_chat_model(self, query: str, model_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        async with self.rate_limiter:
            return await self.agent.chat_model.invoke(query, model_name, params)

    async def process_knowledge_tool(self, query: str, params: Dict[str, Any]) -> Dict[str, Any]:
        # Implement knowledge tool processing logic here
        pass

    async def process_memory_tool(self, query: str, params: Dict[str, Any]) -> Dict[str, Any]:
        # Implement memory tool processing logic here
        pass

    async def process_online_knowledge_tool(self, query: str, params: Dict[str, Any]) -> Dict[str, Any]:
        # Implement online knowledge tool processing logic here
        pass
