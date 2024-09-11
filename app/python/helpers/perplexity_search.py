import os
from dotenv import load_dotenv
import logging
from openai import OpenAI
from typing import List, Dict, Union

load_dotenv()

PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")
logger = logging.getLogger(__name__)


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
                messages=[{"role": msg["role"], "content": msg["content"]} for msg in messages],
                stream=True,
                max_tokens=1024,
                timeout=timeout,
            )
            return [
                chunk.choices[0].delta.content
                for chunk in response_stream
                if chunk.choices[0].delta and chunk.choices[0].delta.content is not None
            ]
        else:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": msg["role"], "content": msg["content"]} for msg in messages],
                max_tokens=1024,
                timeout=timeout
            )
            return response.choices[0].message.content if response.choices else ""
    except Exception as e:
        logger.error(f"Error in Perplexity search: {str(e)}")
        return f"An error occurred while performing the Perplexity search: {str(e)}"


def select_sonar_model(complexity: float) -> str:
    # sourcery skip: avoid-function-declarations-in-blocks
    if complexity < 0.3:
        return "llama-3-sonar-small-32k-online"
    elif complexity < 0.7:
        return "llama-3-sonar-medium-32k-online"
    else:
        return "llama-3-sonar-large-32k-online"
