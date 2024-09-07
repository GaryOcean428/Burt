from python.helpers import perplexity_search
from python.helpers import duckduckgo_search
from typing import Dict, Any


def process_question(question: str, config: Dict[str, Any]) -> str:
    perplexity_api_key = config.get("PERPLEXITY_API_KEY")

    if perplexity_api_key:
        try:
            return perplexity_search.search(question, perplexity_api_key)
        except Exception as e:
            print(f"Perplexity search failed: {e}")

    # Fallback to DuckDuckGo if Perplexity fails or API key is not available
    return duckduckgo_search.search(question)
