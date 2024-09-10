import os
import requests
from dotenv import load_dotenv
from typing import Dict, Any
import logging

load_dotenv()

PERPLEXITY_API_KEY = os.getenv("API_KEY_PERPLEXITY")

def perplexity_search(query: str, max_results: int = 5, api_key: str = None, complexity: float = 0.5) -> str:
    if not api_key:
        api_key = PERPLEXITY_API_KEY

    if not api_key:
        return "Perplexity search is not available (API key not set). Falling back to DuckDuckGo search."

    url = "https://api.perplexity.ai/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    model = select_sonar_model(complexity)

    data = {
        "model": model,
        "messages": [{"role": "user", "content": query}],
        "max_tokens": 1024,
    }

    try:
        response = requests.post(url, json=data, headers=headers)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except requests.RequestException as e:
        logger.error(f"Error in Perplexity search: {str(e)}")
        return f"Error in Perplexity search. Falling back to DuckDuckGo search. Error: {str(e)}"

def select_sonar_model(complexity: float) -> str:
    if complexity < 0.3:
        return "sonar-small-online"
    elif complexity < 0.7:
        return "sonar-medium-online"
    else:
        return "sonar-large-online"
