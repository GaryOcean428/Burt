import os
from dotenv import load_dotenv
import logging
from openai import OpenAI
from typing import List, Dict, Union
from .redis_cache import RedisCache
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
import json

load_dotenv()

PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")
logger = logging.getLogger(__name__)

# Retry decorator for API calls
api_retry = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((Exception,)),
    before_sleep=lambda retry_state: logger.info(
        f"Retrying Perplexity API call: attempt {retry_state.attempt_number}"
    ),
)


@api_retry
def perplexity_search(
    query: str,
    max_results: int = 5,
    api_key: str | None = None,
    complexity: float = 0.5,
    timeout: int = 30,
    stream: bool = False,
) -> Union[str, List[Dict[str, str]]]:
    if not api_key:
        api_key = PERPLEXITY_API_KEY

    if not api_key:
        logger.warning(
            "Perplexity API key not set. Falling back to alternative search method."
        )
        return "Perplexity search is not available (API key not set). Falling back to alternative search method."

    # Check cache first
    cache_key = f"perplexity_search:{query}:{max_results}:{complexity}"
    cached_result = RedisCache.get(cache_key)
    if cached_result:
        logger.info(
            f"Retrieved cached Perplexity search result for query: {query[:50]}..."
        )
        return json.loads(cached_result)

    client = OpenAI(api_key=api_key, base_url="https://api.perplexity.ai")

    model = select_sonar_model(complexity)

    messages = [
        {
            "role": "system",
            "content": "You are a helpful AI assistant providing accurate and relevant information.",
        },
        {
            "role": "user",
            "content": f"Provide up to {max_results} relevant results for the query: {query}",
        },
    ]

    try:
        if stream:
            response_stream = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": msg["role"], "content": msg["content"]}
                    for msg in messages
                ],
                stream=True,
                max_tokens=1024,
                timeout=timeout,
            )
            result = [
                chunk.choices[0].delta.content
                for chunk in response_stream
                if chunk.choices[0].delta
                and chunk.choices[0].delta.content is not None
            ]
        else:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": msg["role"], "content": msg["content"]}
                    for msg in messages
                ],
                max_tokens=1024,
                timeout=timeout,
            )
            result = (
                response.choices[0].message.content if response.choices else ""
            )

        # Cache the result
        RedisCache.set(
            cache_key, json.dumps(result), expiration=3600
        )  # Cache for 1 hour

        return result
    except Exception as e:
        logger.error(f"Error in Perplexity search: {str(e)}", exc_info=True)
        return f"An error occurred while performing the Perplexity search: {str(e)}"


def select_sonar_model(complexity: float) -> str:
    if complexity < 0.3:
        return "llama-3-sonar-small-32k-online"
    elif complexity < 0.7:
        return "llama-3-sonar-medium-32k-online"
    else:
        return "llama-3-sonar-large-32k-online"


def assess_complexity(query: str) -> float:
    # This method is now consistent with the one in RAGSystem
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    import nltk

    nltk.download("punkt", quiet=True)
    nltk.download("stopwords", quiet=True)

    # Tokenize the query
    tokens = word_tokenize(query.lower())

    # Remove stopwords
    stop_words = set(stopwords.words("english"))
    tokens = [word for word in tokens if word not in stop_words]

    # Calculate lexical diversity
    lexical_diversity = len(set(tokens)) / len(tokens) if tokens else 0

    # Calculate average word length
    avg_word_length = (
        sum(len(word) for word in tokens) / len(tokens) if tokens else 0
    )

    # Count special characters and numbers
    special_chars = sum(
        1
        for char in query
        if not char.isalnum() and char not in [" ", ".", ",", "!", "?"]
    )
    numbers = sum(1 for char in query if char.isdigit())

    # Calculate complexity score
    complexity = (
        (len(tokens) / 100) * 0.3  # Length factor
        + lexical_diversity * 0.3  # Vocabulary richness
        + (avg_word_length / 10) * 0.2  # Word complexity
        + (special_chars / len(query)) * 0.1  # Special character density
        + (numbers / len(query)) * 0.1  # Number density
    )

    return min(complexity, 1.0)
