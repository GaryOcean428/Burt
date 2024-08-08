MODELS = {
    "1": {"name": "sonnet-3.5", "id": "claude-3-5-sonnet-20240620"},
    "2": {"name": "opus-3", "id": "claude-3-opus-20240229"},
    "3": {"name": "sonnet-3", "id": "claude-3-sonnet-20240229"},
    "4": {"name": "haiku-3", "id": "claude-3-haiku-20240307"},
    "5": {"name": "chatgpt4o", "id": "chatgpt-4-o"},
    "6": {"name": "llama3.1-8b", "id": "llama-3-1-8b"},
    "7": {"name": "llama3.1-70b", "id": "llama-3-1-70b"},
    "8": {"name": "llama3.1-405b", "id": "llama-3-1-405b-groq"},
}

def get_model_info(model_key):
    return MODELS.get(model_key)

def get_model_list():
    return MODELS
