from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain_groq import ChatGroq
import logging

# Load environment variables
load_dotenv()

# Configuration
DEFAULT_TEMPERATURE = 0.7


def get_model_list():
    return [
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
    ]


def get_chat_model(model_name_or_instance, temperature=DEFAULT_TEMPERATURE):
    logging.info(
        "get_chat_model called with model_name_or_instance: "
        f"{model_name_or_instance}"
    )

    if isinstance(model_name_or_instance,
                  (ChatOpenAI, ChatAnthropic, ChatGroq)):
        logging.info("Returning existing model instance")
        return model_name_or_instance

    if not isinstance(model_name_or_instance, str):
        logging.error(
            f"Unsupported chat model type: {type(model_name_or_instance)}"
        )
        raise ValueError(
            f"Unsupported chat model type: {type(model_name_or_instance)}"
        )

    model_name = model_name_or_instance
    logging.info(f"Creating new model instance for: {model_name}")

    try:
        if model_name.startswith("gpt-"):
            return ChatOpenAI(model=model_name, temperature=temperature)
        elif model_name.startswith("claude-"):
            return ChatAnthropic(model=model_name, temperature=temperature)
        elif (
            model_name.startswith("llama-") or
            model_name.startswith("llama3-groq-")
        ):
            return ChatGroq(model=model_name, temperature=temperature)
        else:
            logging.error(f"Unsupported chat model: {model_name}")
            raise ValueError(f"Unsupported chat model: {model_name}")
    except Exception as e:
        logging.error(
            f"Error creating model instance for {model_name}: {str(e)}"
        )
        raise ValueError(
            f"Error creating model instance for {model_name}: {str(e)}"
        ) from e


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
    if model_name in {
        "text-embedding-ada-002",
        "text-embedding-3-small",
        "text-embedding-3-large",
    }:
        return OpenAIEmbeddings(model=model_name)
    else:
        raise ValueError(f"Unsupported embedding model: {model_name}")


# Example usage
# model = get_chat_model("gpt-4o", temperature=0.7)
# response = model.invoke("Your prompt here")
