from typing import Dict, Any, Optional, List, TypedDict, Tuple
import json
import re
from app.python.helpers.tool import Tool
from app.models import get_chat_model, get_embedding_model
from app.python.helpers.vdb import VectorDB
from app.python.helpers.message import HumanMessage, SystemMessage, AIMessage


class MessageDict(TypedDict):
    role: str
    content: str


class AgentConfig:
    chat_model: str
    embeddings_model: str
    pinecone_api_key: str
    pinecone_environment: str
    pinecone_index_name: str
    pinecone_dimension: int
    pinecone_cloud: str
    perplexity_api_key: str

    def __init__(self, **kwargs: Any):
        self.__dict__.update(kwargs)


class Agent:
    def __init__(self, agent_id: int, config: AgentConfig):
        self.id: int = agent_id
        self.config: AgentConfig = config
        self.chat_model: Optional[Any] = None
        self.embedding_model: Optional[Any] = None
        self.tools: Dict[str, Tool] = {}
        self.conversation_history: List[MessageDict] = []
        self.vector_db: Optional[VectorDB] = None
        self.intervention_status: bool = False
        self.data: Dict[str, Any] = {}

    def set_tools(self, tools: Dict[str, Tool]) -> None:
        self.tools = tools

    def get_tools(self) -> Dict[str, Tool]:
        return self.tools

    def initialize_models(self) -> None:
        if not self.chat_model:
            self.chat_model = get_chat_model(self.config.chat_model)
        if not self.embedding_model:
            self.embedding_model = get_embedding_model(
                self.config.embeddings_model
            )
        self.vector_db = VectorDB(self.config.__dict__)

    async def process(
        self, input_text: str, model_name: str, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        if not self.chat_model:
            self.initialize_models()

        chat_model = get_chat_model(model_name)

        self.conversation_history.append(
            {"role": "user", "content": input_text}
        )

        while True:
            if not chat_model:
                raise ValueError("Chat model is not initialized")

            # Convert conversation history to LangChain message format
            messages = [
                (
                    HumanMessage(content=msg["content"])
                    if msg["role"] == "user"
                    else (
                        AIMessage(content=msg["content"])
                        if msg["role"] == "assistant"
                        else SystemMessage(content=msg["content"])
                    )
                )
                for msg in self.conversation_history
            ]

            # Add a system message to ensure the model knows its identity
            system_message = SystemMessage(
                content=f"You are an AI assistant based on the {model_name} model. You do not have real-time information or the ability to browse the internet. Your knowledge is based on your training data. When asked about current events or to access tools, explain your limitations politely."
            )
            messages.insert(0, system_message)

            response = await chat_model.ainvoke(messages)

            if isinstance(response, dict):
                response_content = response.get("content", "")
            elif hasattr(response, "content"):
                response_content = response.content
            else:
                response_content = str(response)

            if tool_call := self.extract_tool_call(str(response_content)):
                tool_name, tool_args = tool_call
                if tool_name in self.tools:
                    tool_result = self.tools[tool_name].execute(**tool_args)
                    self.conversation_history.append(
                        {"role": "assistant", "content": str(response_content)}
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
                    {"role": "assistant", "content": str(response_content)}
                )
                return {
                    "content": response_content,
                    "model_used": model_name,
                    "conversation_history": self.conversation_history,
                }

    def extract_tool_call(
        self, text: str
    ) -> Optional[Tuple[str, Dict[str, Any]]]:
        pattern = r"\[TOOL_CALL\](.*?)\[/TOOL_CALL\]"
        if match := re.search(pattern, text, re.DOTALL):
            tool_call_json = match[1]
            try:
                tool_call = json.loads(tool_call_json)
                return tool_call["name"], tool_call["args"]
            except json.JSONDecodeError:
                return None
        return None

    def set_intervention_status(self, status: bool):
        self.intervention_status = status

    def get_intervention_status(self) -> bool:
        return self.intervention_status

    def set_data_item(self, key: str, value: Any):
        self.data[key] = value

    def get_data_item(self, key: str) -> Any:
        return self.data.get(key)

    def save_state(self, filename: str):
        with open(filename, "w") as file:
            json.dump(self.data, file)

    def load_state(self, filename: str):
        with open(filename, "r") as file:
            self.data = json.load(file)

    def get_history(self):
        return self.conversation_history


def create_agent(agent_id: int, config: Dict[str, Any]) -> Agent:
    agent_config = AgentConfig(**config)
    return Agent(agent_id, agent_config)
