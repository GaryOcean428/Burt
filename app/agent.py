from dataclasses import dataclass, field
import importlib
import inspect
import os
import re
from typing import Any
from app.python.helpers import rate_limiter, files
from langchain_community.chat_models import BaseChatModel
from langchain_community.llms import BaseLLM
from langchain.embeddings.base import Embeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from typing import List, Any

# We'll define Tool as a type alias for now
Tool = Any


@dataclass
class AgentConfig:
    chat_model: BaseChatModel | BaseLLM
    utility_model: BaseChatModel | BaseLLM
    backup_utility_model: BaseChatModel | BaseLLM
    embeddings_model: Embeddings
    memory_subdir: str = ""
    auto_memory_count: int = 3
    auto_memory_skip: int = 2
    rate_limit_seconds: int = 60
    rate_limit_requests: int = 120
    rate_limit_input_tokens: int = 200000
    rate_limit_output_tokens: int = 200000
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
    def __init__(self, number: int, config: AgentConfig):
        # agent config
        self.config = config

        # non-config vars
        self.number = number
        self.agent_name = f"Agent {self.number}"

        try:
            system_prompt_template = files.read_file("./prompts/agent.system.md")
            self.system_prompt = self.format_template(
                system_prompt_template, agent_name=self.agent_name
            )
        except FileNotFoundError:
            print(
                "Warning: ./prompts/agent.system.md not found. Using default system prompt."
            )
            self.system_prompt = f"You are {self.agent_name}, an AI assistant designed to help users with various tasks."

        try:
            self.tools_prompt = files.read_file("./prompts/agent.tools.md")
        except FileNotFoundError:
            print(
                "Warning: ./prompts/agent.tools.md not found. Tools prompt will be empty."
            )
            self.tools_prompt = ""

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
        self.load_tools()

    def format_template(self, template: str, **kwargs) -> str:
        def replace(match):
            key = match.group(1)
            return str(kwargs.get(key, match.group(0)))

        return re.sub(r"\{\{(\w+)\}\}", replace, template)

    def create_planner(self):
        planner_prompt = PromptTemplate(
            input_variables=["objective", "knowledge"],
            template="Given the objective: {objective}\n"
            "And the following knowledge: {knowledge}\n"
            "Create a step-by-step plan to achieve this objective. "
            "Consider using available tools and sub-agents if necessary.",
        )
        return RunnableSequence(planner_prompt | self.config.utility_model)

    def create_executor(self):
        executor_prompt = PromptTemplate(
            input_variables=["plan", "step"],
            template="Given the plan: {plan}\n"
            "Execute the following step: {step}\n"
            "Provide the result of this step's execution.",
        )
        return RunnableSequence(executor_prompt | self.config.chat_model)

    def process(
        self, user_input: str, selected_model: str, params: Dict[str, Any]
    ) -> str:
        try:
            if not self.rate_limiter.check_and_update():
                return (
                    "I'm sorry, but I've reached my rate limit. Please try again later."
                )

            if self.intervention_status:
                return self.handle_intervention(user_input)

            # Generate a plan using the planner
            knowledge = self.use_tool("knowledge_tool", {"question": user_input})
            plan = self.planner.invoke(
                {"objective": user_input, "knowledge": knowledge}
            )

            # Parse the plan into steps
            steps = self.parse_plan(plan)

            # Execute each step of the plan
            results = []
            for step in steps:
                if "code_execution" in step.lower():
                    result = self.use_tool(
                        "code_execution_tool", {"runtime": "python", "code": step}
                    )
                elif "helper_agents" in step.lower():
                    result = self.use_tool("call_helper_agents", {"task": step})
                else:
                    result = self.executor.invoke(
                        {
                            "plan": plan,
                            "step": step,
                            "model": selected_model,
                            "params": params,
                        }
                    )
                results.append(result)

            # Combine the results into a final response
            final_response = self.combine_results(results)

            # Update conversation history and memory
            self.update_history(user_input, final_response)
            self.use_tool("memory_tool", {"action": "save", "text": final_response})

            return final_response
        except Exception as e:
            error_message = f"An error occurred while processing the input: {str(e)}"
            self.update_history(user_input, error_message)
            return error_message

    def use_tool(self, tool_name: str, args: dict) -> str:
        tool = self.tools.get(tool_name)
        if tool is None:
            raise ValueError(f"Tool '{tool_name}' not found")

        response = tool.execute(**args)
        return response.message

    def parse_plan(self, plan: str) -> List[str]:
        return [step.strip() for step in plan.split("\n") if step.strip()]

    def combine_results(self, results: List[str]) -> str:
        return "\n".join(results)

    def update_history(self, user_input: str, agent_response: str):
        self.history.append({"role": "user", "content": user_input})
        self.history.append({"role": "assistant", "content": agent_response})

        if len(self.history) > self.config.msgs_keep_max:
            keep_start = self.config.msgs_keep_start
            keep_end = self.config.msgs_keep_end
            self.history = self.history[:keep_start] + self.history[-keep_end:]

    def load_tools(self):
        self.tools = {}
        tools_dir = os.path.join(os.path.dirname(__file__), "python", "tools")
        for filename in os.listdir(tools_dir):
            if filename.endswith(".py") and not filename.startswith("__"):
                module_name = filename[:-3]
                module = importlib.import_module(f"app.python.tools.{module_name}")
                for _, obj in inspect.getmembers(module):
                    if inspect.isclass(obj) and issubclass(obj, Tool) and obj != Tool:
                        tool_instance = obj(self)
                        self.tools[tool_instance.name] = tool_instance

    def handle_intervention(self, user_input: str) -> str:
        response = f"Intervention received: {self.intervention_message}\n"
        response += f"User input: {user_input}\n"
        response += "Processing intervention..."
        # Add logic here to handle the intervention based on the user_input
        self.intervention_status = False
        self.intervention_message = ""
        return response

    def save_state(self, filepath: str):
        state = {
            "number": self.number,
            "agent_name": self.agent_name,
            "history": self.history,
            "last_message": self.last_message,
            "intervention_message": self.intervention_message,
            "intervention_status": self.intervention_status,
            "data": self.data,
        }
        with open(filepath, "w") as f:
            json.dump(state, f)

    def load_state(self, filepath: str):
        with open(filepath, "r") as f:
            state = json.load(f)
        self.number = state["number"]
        self.agent_name = state["agent_name"]
        self.history = state["history"]
        self.last_message = state["last_message"]
        self.intervention_message = state["intervention_message"]
        self.intervention_status = state["intervention_status"]
        self.data = state["data"]

    # Existing getter and setter methods...

    def get_data_item(self, key: str) -> Any:
        return self.data.get(key)

    def set_data(self, key: str, value: Any) -> None:
        self.data[key] = value

    def get_tools(self):
        return self.tools

    def set_tools(self, tools: Dict[str, Any]) -> None:
        self.tools = tools

    def get_history(self):
        return self.history

    def set_history(self, history: List[Dict[str, str]]) -> None:
        self.history = history

    def get_last_message(self):
        return self.last_message

    def set_last_message(self, last_message: str) -> None:
        self.last_message = last_message

    def get_intervention_message(self):
        return self.intervention_message

    def set_intervention_message(self, intervention_message: str) -> None:
        self.intervention_message = intervention_message

    def get_intervention_status(self):
        return self.intervention_status

    def set_intervention_status(
        self, intervention_status: bool, intervention_message: str
    ) -> None:
        self.intervention_status = intervention_status
        self.intervention_message = intervention_message

    def get_rate_limiter(self):
        return self.rate_limiter

    def set_rate_limiter(self, rate_limiter: rate_limiter.RateLimiter) -> None:
        self.rate_limiter = rate_limiter

    def get_data(self) -> Dict[str, Any]:
        return self.data

    def set_data_item(self, key: str, value: Any) -> None:
        self.data[key] = value

    def get_config(self):
        return self.config

    def set_config(self, config: AgentConfig) -> None:
        self.config = config

    def get_agent_name(self) -> str:
        return self.agent_name

    def set_agent_name(self, agent_name: str) -> None:
        self.agent_name = agent_name

    def get_number(self):
        return self.number

    def set_number(self, number: int) -> None:
        self.number = number

    def get_system_prompt(self):
        return self.system_prompt

    def set_system_prompt(self, system_prompt: str) -> None:
        self.system_prompt = system_prompt

    def get_tools_prompt(self):
        return self.tools_prompt

    def set_tools_prompt(self, tools_prompt: str) -> None:
        self.tools_prompt = tools_prompt
