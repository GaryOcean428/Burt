import os
import time
import logging
from typing import List, Dict, Any
from pinecone import Pinecone, ServerlessSpec
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

index_name = os.getenv("PINECONE_INDEX_NAME")
environment = os.getenv("PINECONE_ENVIRONMENT")
dimension = int(os.getenv("PINECONE_DIMENSION", "1536"))
cloud = os.getenv("PINECONE_CLOUD", "aws")

# Retry decorator
pinecone_retry = retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    retry=retry_if_exception_type((Exception,)),
    before_sleep=lambda retry_state: logger.info(
        f"Retrying Pinecone operation: attempt {retry_state.attempt_number}"
    ),
)


# Check if the index exists, if not create it
@pinecone_retry
def create_index_if_not_exists():
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud=cloud, region=environment),
        )
        logger.info(f"Created Pinecone index: {index_name}")
    else:
        logger.info(f"Pinecone index {index_name} already exists")


create_index_if_not_exists()

# Get the index
index = pc.Index(index_name)


@pinecone_retry
def upsert_vectors(vectors: List[Dict[str, Any]]):
    try:
        result = index.upsert(vectors=vectors)
        logger.info(f"Successfully upserted {len(vectors)} vectors")
        return result
    except Exception as e:
        logger.error(f"Error upserting vectors: {str(e)}")
        raise


@pinecone_retry
def query_vectors(query_vector: List[float], top_k: int = 5):
    try:
        result = index.query(vector=query_vector, top_k=top_k)
        logger.info(f"Successfully queried vectors with top_k={top_k}")
        return result
    except Exception as e:
        logger.error(f"Error querying vectors: {str(e)}")
        raise


@pinecone_retry
def delete_vectors(ids: List[str]):
    try:
        result = index.delete(ids=ids)
        logger.info(f"Successfully deleted {len(ids)} vectors")
        return result
    except Exception as e:
        logger.error(f"Error deleting vectors: {str(e)}")
        raise


# Health check function
def check_pinecone_health():
    try:
        # Perform a simple query to check if Pinecone is responsive
        query_vectors([0] * dimension, top_k=1)
        logger.info("Pinecone health check passed")
        return True
    except Exception as e:
        logger.error(f"Pinecone health check failed: {str(e)}")
        return False


# Initialization
logger.info("Initializing Pinecone connection...")
if check_pinecone_health():
    logger.info("Pinecone connection established successfully")
else:
    logger.warning(
        "Failed to establish Pinecone connection. Some features may not work properly."
    )
