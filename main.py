"""Main script for initializing and running the AI agent."""

import sys
import os
from dotenv import load_dotenv
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
import threading
import atexit
import signal
from typing import Dict, List, Optional, Union, Any
from app.agent import Agent, AgentConfig
from python.tools.memory_tool import initialize as init_memory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.chat_models import ChatOllama
from langchain_core.language_models import BaseLanguageModel
from langchain_core.embeddings import Embeddings
from langchain.agents import initialize_agent, Tool, AgentType
from pydantic import SecretStr
from langchain_community.vectorstores import FAISS
from app.config import load_config
from app.models import get_model, get_model_list
from langchain.chat_models import BaseChatModel
from langchain_core.language_models.llms import BaseLLM

# Load environment variables from .env file
load_dotenv()

# Instantiate the AdvancedRouter
router = AdvancedRouter()

def capture_keys():
    # Implementation of capture_keys function
    pass

def save_memory(agent):
    # Implementation of save_memory function
    pass

def signal_handler(signum, frame):
    # Implementation of signal_handler function
    pass

def get_model_instance(model_name: str) -> BaseChatModel | BaseLLM:
    model = get_model(model_name)
    if isinstance(model, (BaseChatModel, BaseLLM)):
        return model
    raise TypeError(f"Model {model_name} is not a BaseChatModel or BaseLLM")

def chat(agent):
    print(f"Starting chat with {agent.agent_name}...")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Exiting chat...")
            break
        model_name = router.select_model(user_input)
        params = router.adjust_parameters(model_name, user_input)
        agent.config.chat_model = get_model(model_name, **params)
        response = agent.process(user_input)
        print(f"{agent.agent_name}: {response}")

if __name__ == "__main__":
    print("Initializing framework...")

    threading.Thread(target=capture_keys, daemon=True).start()

    # Create model instances
    chat_model_name = "llama-3.1-8b"  # Use Groq llama3.1-8b as the base model
    chat_model = get_model_instance(chat_model_name)
    utility_model = get_model_instance("claude-3.5-sonnet")  # Use Claude 3.5 Sonnet as the utility model
    embeddings_model = OpenAIEmbeddings()

    # Create an AgentConfig instance with the required models
    config = AgentConfig(
        chat_model=get_model_instance(chat_model_name),
        utility_model=get_model_instance("claude-3.5-sonnet"),
        embeddings_model=embeddings_model
    )

    # Create the Agent instance with the config
    agent = Agent(number=1, config=config)

    atexit.register(save_memory, agent=agent)
    signal.signal(signal.SIGINT, signal_handler)

    chat(agent)
