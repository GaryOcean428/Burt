from typing import Dict, Any, Optional, List, TypedDict, Tuple
import json
import re
from app.python.helpers.tool import Tool
from app.models import get_chat_model


class MessageDict(TypedDict):
    role: str
    content: str


class AgentConfig:
    chat_model: str
    embeddings_model: str

    def __init__(self, **kwargs: Any):
        self.__dict__.update(kwargs)


class Agent:
    def __init__(self, id: int, config: AgentConfig):
        self.id: int = id
        self.config: AgentConfig = config
        self.chat_model: Optional[Any] = None
        self.embedding_model: Optional[Any] = None
        self.tools: Dict[str, Tool] = {}
        self.conversation_history: List[MessageDict] = []

    def set_tools(self, tools: Dict[str, Tool]) -> None:
        self.tools = tools

    def get_tools(self) -> Dict[str, Tool]:
        return self.tools

    def initialize_models(self) -> None:
        from app.models import get_embedding_model

        self.chat_model = get_chat_model(self.config.chat_model)
        if self.chat_model is None:
            raise ValueError(
                f"Failed to initialize chat model: {self.config.chat_model}"
            )
        self.embedding_model = get_embedding_model(self.config.embeddings_model)

    def process(self, input_text: str, model_name: str, params: Dict[str, Any]) -> str:
        if not self.chat_model:
            self.initialize_models()

        chat_model = (
            self.chat_model
            if model_name == self.config.chat_model
            else get_chat_model(model_name)
        )

        self.conversation_history.append({"role": "user", "content": input_text})
        while True:
            if not self.chat_model:
                raise ValueError("Chat model is not initialized")
            response = self.chat_model(self.conversation_history, **params)
            response_content: str = (
                response.content if isinstance(response, dict) else str(response)
            )

            if tool_call := self.extract_tool_call(response_content):
                tool_name, tool_args = tool_call
                if tool_name in self.tools:
                    tool_result = self.tools[tool_name].run(**tool_args)
                    self.conversation_history.append(
                        {"role": "assistant", "content": response_content}
                    )
                    self.conversation_history.append(
                        {
                            "role": "system",
                            "content": f"Tool {tool_name} returned: {tool_result}",
                        }
                    )
                else:
                    self.conversation_history.append(
                        {
                            "role": "system",
                            "content": f"Error: Tool {tool_name} not found.",
                        }
                    )
            else:
                self.conversation_history.append(
                    {"role": "assistant", "content": response_content}
                )
                return response_content

    def extract_tool_call(self, text: str) -> Optional[Tuple[str, Dict[str, Any]]]:
        pattern = r"\[TOOL_CALL\](.*?)\[/TOOL_CALL\]"
        if match := re.search(pattern, text, re.DOTALL):
            tool_call_json = match[1]
            try:
                tool_call = json.loads(tool_call_json)
                return tool_call["name"], tool_call["args"]
            except json.JSONDecodeError:
                return None
        return None


def create_agent(id: int, config: Dict[str, Any]) -> Agent:
    agent_config = AgentConfig(**config)
    return Agent(id, agent_config)
