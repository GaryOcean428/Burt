import os
from dotenv import load_dotenv
from langchain_openai import (
    ChatOpenAI,
    OpenAI,
    OpenAIEmbeddings,
    AzureChatOpenAI,
    AzureOpenAIEmbeddings,
    AzureOpenAI,
)
from langchain_community.llms.ollama import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatAnthropic, ChatPerplexity
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    HarmBlockThreshold,
    HarmCategory,
)
from groq import Groq
from langchain_core.language_models import BaseLLM
from langchain_core.language_models.chat_models import BaseChatModel

# Load environment variables
load_dotenv()

# Configuration
DEFAULT_TEMPERATURE = 0.7


def get_api_key(service):
    return os.getenv(f"API_KEY_{service.upper()}") or os.getenv(
        f"{service.upper()}_API_KEY"
    )


def get_ollama_chat(
    model_name: str, temperature=DEFAULT_TEMPERATURE, base_url="http://localhost:11434"
):
    return Ollama(model=model_name, temperature=temperature, base_url=base_url)


def get_ollama_embedding(model_name: str, temperature=DEFAULT_TEMPERATURE):
    return OllamaEmbeddings(model=model_name, temperature=temperature)


def get_huggingface_embedding(model_name: str):
    return HuggingFaceEmbeddings(model_name=model_name)


def get_lmstudio_chat(
    model_name: str,
    base_url="http://localhost:1234/v1",
    temperature=DEFAULT_TEMPERATURE,
):
    return ChatOpenAI(
        model=model_name, base_url=base_url, temperature=temperature, api_key=None
    )


def get_lmstudio_embedding(model_name: str, base_url="http://localhost:1234/v1"):
    return OpenAIEmbeddings(model=model_name, base_url=base_url, api_key=None)


def get_anthropic_chat(model_name: str, api_key=None, temperature=DEFAULT_TEMPERATURE):
    api_key = api_key or get_api_key("anthropic")
    return ChatAnthropic(
        model_name=model_name,
        temperature=temperature,
        api_key=api_key,
        timeout=600,
        stop=None,
        base_url=None,
    )


def get_openai_chat(model_name: str, api_key=None, temperature=DEFAULT_TEMPERATURE):
    api_key = api_key or get_api_key("openai")
    return ChatOpenAI(model_name=model_name, temperature=temperature, api_key=api_key)


def get_openai_instruct(model_name: str, api_key=None, temperature=DEFAULT_TEMPERATURE):
    api_key = api_key or get_api_key("openai")
    return OpenAI(model=model_name, temperature=temperature, api_key=api_key)


def get_openai_embedding(model_name: str, api_key=None):
    api_key = api_key or get_api_key("openai")
    return OpenAIEmbeddings(model=model_name, api_key=api_key)


def get_azure_openai_chat(
    deployment_name: str,
    api_key=None,
    temperature=DEFAULT_TEMPERATURE,
    azure_endpoint=None,
):
    api_key = api_key or get_api_key("openai_azure")
    azure_endpoint = azure_endpoint or os.getenv("OPENAI_AZURE_ENDPOINT")
    return AzureChatOpenAI(
        model=deployment_name,
        temperature=temperature,
        api_key=api_key,
        azure_endpoint=azure_endpoint,
    )


def get_azure_openai_instruct(
    deployment_name: str,
    api_key=None,
    temperature=DEFAULT_TEMPERATURE,
    azure_endpoint=None,
):
    api_key = api_key or get_api_key("openai_azure")
    azure_endpoint = azure_endpoint or os.getenv("OPENAI_AZURE_ENDPOINT")
    return AzureOpenAI(
        model=deployment_name,
        temperature=temperature,
        api_key=api_key,
        azure_endpoint=azure_endpoint,
    )


def get_azure_openai_embedding(deployment_name: str, api_key=None, azure_endpoint=None):
    api_key = api_key or get_api_key("openai_azure")
    azure_endpoint = azure_endpoint or os.getenv("OPENAI_AZURE_ENDPOINT")
    return AzureOpenAIEmbeddings(
        model=deployment_name, api_key=api_key, azure_endpoint=azure_endpoint
    )


def get_google_chat(model_name: str, api_key=None, temperature=DEFAULT_TEMPERATURE):
    api_key = api_key or get_api_key("google")
    return ChatGoogleGenerativeAI(
        model=model_name,
        temperature=temperature,
        api_key=api_key,
        safety_settings={
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE
        },
    )


def get_groq_chat(model_name, api_key, temperature=0.7, **kwargs):
    groq = Groq(api_key=api_key)
    max_new_tokens = kwargs.get("max_new_tokens", 2048)

    def chat_completion(messages):
        response = groq.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_new_tokens,
        )
        return response.choices[0].message.content

    return chat_completion


def get_openrouter(
    model_name: str = "meta-llama/llama-3.1-8b-instruct:free",
    api_key=None,
    temperature=DEFAULT_TEMPERATURE,
):
    api_key = api_key or get_api_key("openrouter")
    return ChatOpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        model=model_name,
        temperature=temperature,
    )


def get_embedding_hf(model_name="sentence-transformers/all-MiniLM-L6-v2"):
    return HuggingFaceEmbeddings(model_name=model_name)


def get_embedding_openai(api_key=None):
    api_key = api_key or get_api_key("openai")
    return OpenAIEmbeddings(openai_api_key=api_key)


def get_perplexity_chat(
    model_name: str = "sonar-medium-chat", api_key=None, temperature=DEFAULT_TEMPERATURE
):
    api_key = api_key or get_api_key("perplexity")
    return ChatPerplexity(model=model_name, api_key=api_key, temperature=temperature)


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


def get_model(model_name: str) -> BaseChatModel | BaseLLM:
    if model_name.startswith("gpt-4"):
        return get_openai_chat(
            model_name, api_key=None, temperature=DEFAULT_TEMPERATURE
        )
    elif model_name.startswith("claude"):
        return get_anthropic_chat(
            model_name, api_key=None, temperature=DEFAULT_TEMPERATURE
        )
    elif model_name.startswith("llama-3.1"):
        return get_groq_chat(
            model_name, api_key=get_api_key("groq"), temperature=DEFAULT_TEMPERATURE
        )
    elif model_name.startswith("gemini"):
        return get_google_chat(
            model_name, api_key=None, temperature=DEFAULT_TEMPERATURE
        )
    elif model_name.startswith("mistral") or model_name.startswith("mixtral"):
        return get_mistral_chat(
            model_name, api_key=None, temperature=DEFAULT_TEMPERATURE
        )
    elif model_name in ["command-r-plus", "reka-core"]:
        return get_openrouter_chat(
            model_name, api_key=None, temperature=DEFAULT_TEMPERATURE
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")


# Example usage
# model = get_model("gpt-4o", temperature=0.7)
# response = model.invoke("Your prompt here")
