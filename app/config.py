import os
from dotenv import load_dotenv
from app.models import get_groq_chat, get_openai_chat, get_openai_embedding
from app.agent import AgentConfig

# Load environment variables
load_dotenv()

# Configuration constants
ROUTER_THRESHOLD = 0.7


def load_config():
    # Create an AgentConfig object
    config = AgentConfig(
        chat_model=get_groq_chat("llama-3.1-8b", api_key=os.getenv("API_KEY_GROQ")),
        utility_model=get_groq_chat("llama-3.1-70b", api_key=os.getenv("API_KEY_GROQ")),
        backup_utility_model=get_openai_chat(
            os.getenv("OPENAI_UTILITY_MODEL", "gpt-4o-mini")
        ),
        embeddings_model=get_openai_embedding(
            model_name=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-ada-002")
        ),
        rate_limit_seconds=60,
        rate_limit_requests=120,  # Increased from default
        rate_limit_input_tokens=200000,  # Increased from default
        rate_limit_output_tokens=200000,  # Increased from default
    )
    return config


# Add any other configuration variables as needed
