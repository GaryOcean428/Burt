from dotenv import load_dotenv
from langchain_openai import (
    ChatOpenAI,
    OpenAIEmbeddings,
)
from langchain_community.llms.ollama import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
)
from langchain_anthropic import ChatAnthropic
from langchain_community.embeddings import HuggingFaceEmbeddings
import logging
import requests

# Load environment variables
load_dotenv()

# Configuration
DEFAULT_TEMPERATURE = 0.7


def is_ollama_available():
    try:
        response = requests.get("http://localhost:11434/api/tags")
        return response.status_code == 200
    except requests.ConnectionError:
        return False


def get_ollama_chat(
    model_name: str, temperature=DEFAULT_TEMPERATURE, base_url="http://localhost:11434"
):
    return Ollama(model=model_name, temperature=temperature, base_url=base_url)


def get_ollama_embedding(model_name: str, base_url="http://localhost:11434"):
    return OllamaEmbeddings(model=model_name, base_url=base_url)


def get_model_list():
    return [
        "gpt-4o",
        "gpt-4o-mini",
        "claude-3-5-sonnet-20240620",
        "claude-3-opus-20240229",
        "claude-3-sonnet-20240229",
        "claude-3-haiku-20240307",
        "llama-3.1-405b",
        "llama-3.1-70b",
        "llama-3.1-8b",
        "llama-3.1-sonar-small-128k-online",
        "llama-3.1-sonar-large-128k-online",
        "llama-3.1-sonar-huge-128k-online",
        "gemini-1.5-pro",
        "gemini-1.5-flash",
        "mistral-large-2",
        "mixtral-8x22b",
        "mistral-nemo",
        "mixtral-8x7b",
        "command-r-plus",
        "reka-core",
        "llama3-groq-70b-8192-tool-use-preview",
        "llama3-groq-8b-8192-tool-use-preview",
    ]


def get_chat_model(model_name_or_instance, temperature=DEFAULT_TEMPERATURE):
    logging.info(
        f"get_chat_model called with model_name_or_instance: {model_name_or_instance}"
    )

    if isinstance(
        model_name_or_instance,
        (ChatOpenAI, ChatAnthropic, Ollama, ChatGoogleGenerativeAI),
    ):
        logging.info("Returning existing model instance")
        return model_name_or_instance

    if not isinstance(model_name_or_instance, str):
        logging.error(f"Unsupported chat model type: {type(model_name_or_instance)}")
        raise ValueError(f"Unsupported chat model type: {type(model_name_or_instance)}")

    model_name = model_name_or_instance
    logging.info(f"Creating new model instance for: {model_name}")

    if model_name.startswith("gpt-"):
        return ChatOpenAI(model=model_name, temperature=temperature)
    elif model_name.startswith("claude-"):
        return ChatAnthropic(model=model_name, temperature=temperature)
    elif model_name.startswith("llama-") or model_name.startswith("llama3-groq-"):
        if not is_ollama_available():
            raise ValueError("Ollama is not available. Please start the Ollama server.")
        return get_ollama_chat(model_name, temperature)
    elif model_name.startswith("gemini-"):
        return ChatGoogleGenerativeAI(model=model_name, temperature=temperature)
    else:
        logging.error(f"Unsupported chat model: {model_name}")
        raise ValueError(f"Unsupported chat model: {model_name}")


def get_tool_use_model(temperature=DEFAULT_TEMPERATURE):
    tool_use_models = [
        "llama3-groq-70b-8192-tool-use-preview",
        "llama3-groq-8b-8192-tool-use-preview",
        "gpt-4o",
        "claude-3-5-sonnet-20240620",
    ]
    for model in tool_use_models:
        try:
            return get_chat_model(model, temperature)
        except Exception:
            continue
    raise ValueError("No available tool use model found")


def get_embedding_model(model_name: str):
    if model_name == "text-embedding-ada-002":
        return OpenAIEmbeddings()
    elif model_name.startswith("llama-"):
        if not is_ollama_available():
            raise ValueError("Ollama is not available. Please start the Ollama server.")
        return get_ollama_embedding(model_name)
    else:
        return HuggingFaceEmbeddings(model_name=model_name)


# Example usage
# model = get_chat_model("gpt-4o", temperature=0.7)
# response = model.invoke("Your prompt here")
