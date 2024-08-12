MODELS = {
    "1": {"name": "sonnet-3.5", "id": "claude-3-5-sonnet-20240620", "intellect": 5, "cost": 5},
    "2": {"name": "opus-3", "id": "claude-3-opus-20240229", "intellect": 4, "cost": 4},
    "3": {"name": "chatgpt4o", "id": "chatgpt-4-o", "intellect": 4, "cost": 4},
    "4": {"name": "llama3.1-405b", "id": "llama-3-1-405b-groq", "intellect": 4, "cost": 2},
    "5": {"name": "llama3.1-70b", "id": "llama-3-1-70b", "intellect": 3, "cost": 2},
    "6": {"name": "llama3.1-8b", "id": "llama-3-1-8b", "intellect": 2, "cost": 1},
}

def get_model_info(model_key):
    return MODELS.get(model_key)

def get_model_list():
    return MODELS
