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

# Load environment variables
load_dotenv()

# Configuration
DEFAULT_TEMPERATURE = 0.7

def get_ollama_chat(model_name: str, temperature=DEFAULT_TEMPERATURE, base_url="http://localhost:11434"):
    return Ollama(model=model_name, temperature=temperature, base_url=base_url)

def get_ollama_embedding(model_name: str, base_url="http://localhost:11434"):
    return OllamaEmbeddings(model=model_name, base_url=base_url)

def get_model_list():
    return [
        "gpt-4o",
        "gpt-4o-mini",
        "claude-3.5-sonnet",
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
    ]

def get_chat_model(model_name: str, temperature=DEFAULT_TEMPERATURE):
    if model_name.startswith("gpt-"):
        return ChatOpenAI(model=model_name, temperature=temperature)
    elif model_name.startswith("claude-"):
        return ChatAnthropic(model=model_name, temperature=temperature)
    elif model_name.startswith("llama-"):
        return get_ollama_chat(model_name, temperature)
    elif model_name.startswith("gemini-"):
        return ChatGoogleGenerativeAI(model=model_name, temperature=temperature)
    else:
        raise ValueError(f"Unsupported chat model: {model_name}")

def get_embedding_model(model_name: str):
    if model_name == "text-embedding-ada-002":
        return OpenAIEmbeddings()
    elif model_name.startswith("llama-"):
        return get_ollama_embedding(model_name)
    else:
        return HuggingFaceEmbeddings(model_name=model_name)

# Example usage
# model = get_chat_model("gpt-4o", temperature=0.7)
# response = model.invoke("Your prompt here")
