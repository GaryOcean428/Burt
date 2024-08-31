"""Main script for initializing and running the AI agent."""

import sys
import os
import threading
import atexit
import signal
from typing import Dict, List, Optional, Union, Any
from agent import Agent, AgentConfig
from python.tools.memory_tool import initialize as init_memory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.chat_models import ChatOllama
from langchain.schema import BaseLanguageModel
from langchain.embeddings import Embeddings

# Constants
CHAT_MODELS: Dict[str, List[str]] = {
    "OpenAI": ["gpt-3.5-turbo", "gpt-4"],
    "Anthropic": ["claude-2", "claude-instant-1"],
    "Google": ["gemini-pro"],
    "Ollama": ["gemma:7b", "mistral:7b"]
}

EMBEDDING_MODELS: List[str] = ["text-embedding-ada-002"]

MODEL_INSTANCES: Dict[str, Union[BaseLanguageModel, Embeddings]] = {}

def get_model_instance(model_name: str) -> Union[BaseLanguageModel, Embeddings]:
    """Get or create a model instance based on the model name."""
    if model_name in MODEL_INSTANCES:
        return MODEL_INSTANCES[model_name]

    if "gpt" in model_name:
        instance = ChatOpenAI(model=model_name, api_key=os.getenv('API_KEY_OPENAI'))
    elif "claude" in model_name:
        instance = ChatAnthropic(model=model_name, api_key=os.getenv('API_KEY_ANTHROPIC'))
    elif "gemini" in model_name:
        instance = ChatGoogleGenerativeAI(model=model_name, api_key=os.getenv('API_KEY_GOOGLE'))
    elif "gemma" in model_name or "mistral" in model_name.lower():
        instance = ChatOllama(model=model_name)
    elif "text-embedding" in model_name:
        instance = OpenAIEmbeddings(model=model_name, api_key=os.getenv('API_KEY_OPENAI'))
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    MODEL_INSTANCES[model_name] = instance
    return instance

def select_model(model_type: str, available_models: Dict[str, List[str]]) -> str:
    """Prompt the user to select a model from the available options."""
    print(f"\nAvailable {model_type} models:")
    all_models: List[str] = []
    for provider, models_list in available_models.items():
        print(f"\n{provider}:")
        for i, model in enumerate(models_list, start=len(all_models) + 1):
            print(f"{i}. {model}")
        all_models.extend(models_list)

    print(f"\n{len(all_models) + 1}. Enter custom model")

    while True:
        try:
            choice = int(input(f"\nSelect a {model_type} model (1-{len(all_models) + 1}): "))
            if 1 <= choice <= len(all_models):
                return all_models[choice - 1]
            if choice == len(all_models) + 1:
                return input("Enter the name of your custom model: ")
            print("Invalid choice. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a number.")

def initialize() -> Agent:
    """Initialize the AI agent with selected models and configuration."""
    chat_model_name = select_model("chat", CHAT_MODELS)
    embedding_model_name = select_model("embedding", {"Embedding": EMBEDDING_MODELS})

    chat_llm = get_model_instance(chat_model_name)
    utility_llm = chat_llm
    embedding_model = get_model_instance(embedding_model_name)

    if not isinstance(chat_llm, BaseLanguageModel) or not isinstance(embedding_model, Embeddings):
        raise TypeError("Invalid model types")

    print(f"\nSelected models:")
    print(f"Chat model: {chat_model_name}")
    print(f"Utility model: {chat_model_name}")
    print(f"Embedding model: {embedding_model_name}")

    config = AgentConfig(
        llm=chat_llm,
        utility_llm=utility_llm,
        embedding_model=embedding_model,
        auto_memory_count=0,
        code_exec_docker_enabled=True,
        code_exec_ssh_enabled=True,
    )
    print(config)

    agent = Agent(number=0, config=config)
    init_memory(agent)

    print("Initialization successful!")
    return agent

def save_memory(agent: Optional[Agent]) -> None:
    """Save the agent's memory state."""
    if agent:
        print("Saving memory state...")
        # Implement the actual memory saving logic here

def signal_handler(signum: int, frame: Any) -> None:
    """Handle interrupt signals by saving memory and exiting."""
    print("\nInterrupt received, saving memory state and exiting...")
    save_memory(agent)
    sys.exit(0)

def chat(agent: Agent) -> None:
    """Run the chat session with the AI agent."""
    print("\nStarting chat session. Type 'exit' to end the conversation.")
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == "exit":
            print("Ending chat session.")
            break
        try:
            agent.chat(user_input)
        except Exception as e:
            print(f"Error: An exception occurred while processing your input. {str(e)}")

def capture_keys() -> None:
    """Capture key presses for additional functionality."""
    # Implement key capture logic here

if __name__ == "__main__":
    print("Initializing framework...")

    threading.Thread(target=capture_keys, daemon=True).start()

    agent = initialize()

    atexit.register(save_memory, agent=agent)
    signal.signal(signal.SIGINT, signal_handler)

    chat(agent)
