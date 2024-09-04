import os
from dotenv import load_dotenv
from app.models import get_openai_chat, get_openai_embedding
from app.agent import AgentConfig

# Load environment variables
load_dotenv()

# Configuration constants
ROUTER_THRESHOLD = 0.7

def load_config():
    # Create an AgentConfig object
    config = AgentConfig(
        chat_model=get_openai_chat(os.getenv('OPENAI_CHAT_MODEL', 'gpt-3.5-turbo')),
        utility_model=get_openai_chat(os.getenv('OPENAI_UTILITY_MODEL', 'gpt-3.5-turbo')),
        embeddings_model=get_openai_embedding(model_name=os.getenv('OPENAI_EMBEDDING_MODEL', 'text-embedding-ada-002'))
    )
    return config

# Add any other configuration variables as needed
