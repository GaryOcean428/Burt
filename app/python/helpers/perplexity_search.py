import os
import requests
from dotenv import load_dotenv

load_dotenv()

PERPLEXITY_API_KEY = os.getenv("API_KEY_PERPLEXITY")


def perplexity_search(query, max_results=5):
    if not PERPLEXITY_API_KEY:
        return "Perplexity search is not available (API key not set)."

    url = "https://api.perplexity.ai/chat/completions"
    headers = {
        "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
        "Content-Type": "application/json",
    }
    data = {
        "model": "pplx-7b-online",
        "messages": [{"role": "user", "content": query}],
        "max_tokens": 1024,
    }

    try:
        response = requests.post(url, json=data, headers=headers)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except requests.RequestException as e:
        raise Exception(f"Error in Perplexity search: {str(e)}")
