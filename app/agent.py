from dataclasses import dataclass, field
import time, importlib, inspect, os, json, re
from typing import Any, Optional, Dict, List
from app.python.helpers import extract_tools, rate_limiter, files, errors
from app.python.helpers.print_style import PrintStyle
from langchain.schema import AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.language_models.llms import BaseLLM
from langchain_core.embeddings import Embeddings
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence

@dataclass
class AgentConfig:
    chat_model: BaseChatModel | BaseLLM
    utility_model: BaseChatModel | BaseLLM
    embeddings_model: Embeddings
    memory_subdir: str = ""
    auto_memory_count: int = 3
    auto_memory_skip: int = 2
    rate_limit_seconds: int = 60
    rate_limit_requests: int = 15
    rate_limit_input_tokens: int = 1000000
    rate_limit_output_tokens: int = 0
    msgs_keep_max: int = 25
    msgs_keep_start: int = 5
    msgs_keep_end: int = 10
    response_timeout_seconds: int = 60
    max_tool_response_length: int = 3000
    code_exec_docker_enabled: bool = True
    code_exec_docker_name: str = "agent-zero-exe"
    code_exec_docker_image: str = "frdel/agent-zero-exe:latest"
    code_exec_docker_ports: dict[str, int] = field(
        default_factory=lambda: {"22/tcp": 50022}
    )
    code_exec_docker_volumes: dict[str, dict[str, str]] = field(
        default_factory=lambda: {
            files.get_abs_path("work_dir"): {"bind": "/root", "mode": "rw"}
        }
    )
    code_exec_ssh_enabled: bool = True
    code_exec_ssh_addr: str = "localhost"
    code_exec_ssh_port: int = 50022
    code_exec_ssh_user: str = "root"
    code_exec_ssh_pass: str = "toor"
    additional: Dict[str, Any] = field(default_factory=dict)


class Agent:

    paused = False
    streaming_agent = None

    def __init__(self, number: int, config: AgentConfig):

        # agent config
        self.config = config

        # non-config vars
        self.number = number
        self.agent_name = f"Agent {self.number}"

        system_prompt_template = files.read_file("./prompts/agent.system.md")
        self.system_prompt = self.format_template(system_prompt_template, agent_name=self.agent_name)

        self.tools_prompt = files.read_file("./prompts/agent.tools.md")

        self.history = []
        self.last_message = ""
        self.intervention_message = ""
        self.intervention_status = False
        self.rate_limiter = rate_limiter.RateLimiter(
            max_calls=self.config.rate_limit_requests,
            max_input_tokens=self.config.rate_limit_input_tokens,
            max_output_tokens=self.config.rate_limit_output_tokens,
            window_seconds=self.config.rate_limit_seconds,
        )
        self.data = {}  # free data object all the tools can use

        self.planner = self.create_planner()
        self.executor = self.create_executor()

    def format_template(self, template: str, **kwargs) -> str:
        def replace(match):
            key = match.group(1)
            return str(kwargs.get(key, match.group(0)))
        return re.sub(r'\{\{(\w+)\}\}', replace, template)

    def create_planner(self):
        planner_prompt = PromptTemplate(
            input_variables=["objective"],
            template="Given the objective: {objective}\n"
                     "Create a step-by-step plan to achieve this objective. "
                     "Consider using available tools and sub-agents if necessary."
        )
        return RunnableSequence(planner_prompt | self.config.utility_model)

    def create_executor(self):
        executor_prompt = PromptTemplate(
            input_variables=["plan", "step"],
            template="Given the plan: {plan}\n"
                     "Execute the following step: {step}\n"
                     "Provide the result of this step's execution."
        )
        return RunnableSequence(executor_prompt | self.config.chat_model)
    def process(self, user_input: str) -> str:
        if not self.rate_limiter.check_and_update():
            return "I'm sorry, but I've reached my rate limit. Please try again later."

        # Use knowledge tool to gather information
        knowledge = self.use_tool("knowledge_tool", {"question": user_input})

        # Generate a plan using the planner
        plan = self.planner.invoke({"objective": user_input, "knowledge": knowledge})

        # Parse the plan into steps
        steps = self.parse_plan(plan)

        # Execute each step of the plan
        results = []
        for step in steps:
            if "code_execution" in step.lower():
                result = self.use_tool("code_execution_tool", {"runtime": "python", "code": step})
            elif "helper_agents" in step.lower():
                result = self.use_tool("call_helper_agents", {"task": step})
            else:
                result = self.executor.invoke({"plan": plan, "step": step})
            results.append(result)

        # Combine the results into a final response
        final_response = self.combine_results(results)

        # Update conversation history and memory
        self.update_history(user_input, final_response)
        self.use_tool("memory_tool", {"action": "save", "text": final_response})

        return final_response
    def use_tool(self, tool_name: str, args: dict) -> str:
        tool = self.tools.get(tool_name)
        if tool is None:
            raise ValueError(f"Tool '{tool_name}' not found")
        response = tool.execute(**args)
        return response.message

    def parse_plan(self, plan: str) -> List[str]:
        # Simple parsing: split the plan by newlines and remove any empty lines
        return [step.strip() for step in plan.split('\n') if step.strip()]

    def combine_results(self, results: List[str]) -> str:
        # Simple combination: join all results with newlines
        return "\n".join(results)

    def update_history(self, user_input: str, agent_response: str):
        self.history.append({"role": "user", "content": user_input})
        self.history.append({"role": "assistant", "content": agent_response})

        # Trim history if it exceeds the maximum number of messages to keep
        if len(self.history) > self.config.msgs_keep_max:
            keep_start = self.config.msgs_keep_start
            keep_end = self.config.msgs_keep_end
            self.history = self.history[:keep_start] + self.history[-keep_end:]

    def get_data(self, key: str) -> Any:
        return self.data.get(key)

    def set_data(self, key: str, value: Any) -> None:
        self.data[key] = value

    # ... (rest of the Agent class implementation remains unchanged)
