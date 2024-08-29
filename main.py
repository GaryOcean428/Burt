import sys
import os
import threading
import time
import models
from ansio import application_keypad
from ansio.input import InputEvent, get_input_event
from agent import Agent, AgentConfig
from python.helpers.print_style import PrintStyle
from python.helpers.files import read_file
from python.helpers import files
import python.helpers.timed_input as timed_input

print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")
print(f"sys.path: {sys.path}")
print(f"Current working directory: {os.getcwd()}")

input_lock = threading.Lock()
os.chdir(files.get_abs_path("./work_dir"))  # change CWD to work_dir

# Available models
CHAT_MODELS = {
    "OpenAI": ["gpt-4o-mini"],
    "Ollama": ["gemma2:latest"],
    "LMStudio": ["TheBloke/Mistral-7B-Instruct-v0.2-GGUF"],
    "OpenRouter": ["meta-llama/llama-3-8b-instruct:free"],
    "Azure OpenAI": ["gpt-4o-mini"],
    "Anthropic": ["claude-3-5-sonnet-20240620"],
    "Google": ["gemini-1.5-flash"],
    "Groq": ["llama-3.1-70b-versatile", "llama-3.1-70b", "llama-3.1-70b-base"]
}

EMBEDDING_MODELS = ["text-embedding-3-small", "text-embedding-ada-002"]

def select_model(model_type, available_models):
    print(f"\nAvailable {model_type} models:")
    all_models = []
    for provider, models_list in available_models.items():
        print(f"\n{provider}:")
        for i, model in enumerate(models_list, len(all_models) + 1):
            print(f"{i}. {model}")
        all_models.extend(models_list)
    
    print(f"\n{len(all_models) + 1}. Enter custom model")
    
    while True:
        try:
            choice = int(input(f"\nSelect a {model_type} model (1-{len(all_models) + 1}): "))
            if 1 <= choice <= len(all_models):
                return all_models[choice - 1]
            elif choice == len(all_models) + 1:
                return input("Enter the name of your custom model: ")
            else:
                print("Invalid choice. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a number.")

def get_model_instance(model_name):
    if model_name in CHAT_MODELS["OpenAI"]:
        return models.get_openai_chat(model_name=model_name, temperature=0)
    elif model_name in CHAT_MODELS["Ollama"]:
        return models.get_ollama_chat(model_name=model_name, temperature=0)
    elif model_name in CHAT_MODELS["LMStudio"]:
        return models.get_lmstudio_chat(model_name=model_name, temperature=0)
    elif model_name in CHAT_MODELS["OpenRouter"]:
        return models.get_openrouter(model_name=model_name)
    elif model_name in CHAT_MODELS["Azure OpenAI"]:
        return models.get_azure_openai_chat(deployment_name=model_name, temperature=0)
    elif model_name in CHAT_MODELS["Anthropic"]:
        return models.get_anthropic_chat(model_name=model_name, temperature=0)
    elif model_name in CHAT_MODELS["Google"]:
        return models.get_google_chat(model_name=model_name, temperature=0)
    elif model_name in CHAT_MODELS["Groq"]:
        return models.get_groq_chat(model_name=model_name, temperature=0)
    else:
        print(f"Warning: Using default OpenAI chat for custom model {model_name}")
        return models.get_openai_chat(model_name=model_name, temperature=0)

def initialize():
    # Select chat model
    chat_model_name = select_model("chat", CHAT_MODELS)
    
    # Select embedding model
    embedding_model_name = select_model("embedding", {"Embedding": EMBEDDING_MODELS})

    # Get the chat model instance
    chat_llm = get_model_instance(chat_model_name)

    # utility model used for helper functions (cheaper, faster)
    utility_llm = chat_llm  # change if you want to use a different utility model

    # embedding model used for memory
    embedding_llm = models.get_openai_embedding(model_name=embedding_model_name)

    print(f"\nSelected models:")
    print(f"Chat model: {chat_model_name}")
    print(f"Utility model: {chat_model_name}")  # Since it's the same as chat_llm
    print(f"Embedding model: {embedding_model_name}")

    # agent configuration
    config = AgentConfig(
        chat_model=chat_llm,
        utility_model=utility_llm,
        embeddings_model=embedding_llm,
        auto_memory_count=0,
        code_exec_docker_enabled=True,
        code_exec_ssh_enabled=True,
    )

    # create the first agent
    agent = Agent(number=0, config=config)

    print("Initialization successful!")
    return agent  # Return the created agent instead of exiting

# Main conversation loop
def chat(agent: Agent):
    # start the conversation loop
    while True:
        # ask user for message
        with input_lock:
            if timeout := agent.get_data("timeout"):
                PrintStyle(
                    background_color="#6C3483",
                    font_color="white",
                    bold=True,
                    padding=True,
                ).print(
                    f"User message ({timeout}s timeout, 'w' to wait, 'e' to leave):"
                )
                user_input = timeout_input("> ", timeout=timeout)

                if not user_input:
                    user_input = read_file("prompts/fw.msg_timeout.md")
                    PrintStyle(font_color="white", padding=False).stream(
                        f"{user_input}"
                    )
                else:
                    user_input = user_input.strip()
                    if user_input.lower() == "w":  # the user needs more time
                        user_input = input("> ").strip()
                    PrintStyle(font_color="white", padding=False, log_only=True).print(
                        f"> {user_input}"
                    )
            else:
                PrintStyle(
                    background_color="#6C3483",
                    font_color="white",
                    bold=True,
                    padding=True,
                ).print("User message ('e' to leave):")
                user_input = input("> ")
                PrintStyle(font_color="white", padding=False, log_only=True).print(
                    f"> {user_input}"
                )

        # exit the conversation when the user types 'exit'
        if user_input.lower() == "e":
            break

        # send message to agent0,
        assistant_response = agent.message_loop(user_input)

        # print agent0 response
        PrintStyle(
            font_color="white", background_color="#1D8348", bold=True, padding=True
        ).print(f"{agent.agent_name}: response:")
        PrintStyle(font_color="white").print(f"{assistant_response}")

# User intervention during agent streaming
def intervention():
    if not Agent.streaming_agent or Agent.paused:
        return
    Agent.paused = True  # stop agent streaming
    PrintStyle(
        background_color="#6C3483", font_color="white", bold=True, padding=True
    ).print("User intervention ('e' to leave, empty to continue):")

    user_input = input("> ").strip()
    PrintStyle(font_color="white", padding=False, log_only=True).print(
        f"> {user_input}"
    )

    if user_input.lower() == "e":
        os._exit(0)  # exit the conversation when the user types 'exit'
    if user_input:
        Agent.streaming_agent.intervention_message = (
            user_input  # set intervention message if non-empty
        )
    Agent.paused = False  # continue agent streaming

# Capture keyboard input to trigger user intervention
def capture_keys():
    global input_lock
    intervent = False
    while True:
        if intervent:
            intervention()
        intervent = False
        time.sleep(0.1)

        if Agent.streaming_agent:
            with input_lock, application_keypad:
                event: InputEvent | None = get_input_event(timeout=0.1)
                if event and (event.shortcut.isalpha() or event.shortcut.isspace()):
                    intervent = True
                    continue

# User input with timeout
def timeout_input(prompt, timeout=10):
    return timed_input.timeout_input(prompt=prompt, timeout=timeout)

if __name__ == "__main__":
    print("Initializing framework...")

    # Start the key capture thread for user intervention during agent streaming
    threading.Thread(target=capture_keys, daemon=True).start()

    # Start the initialization and get the created agent
    agent = initialize()

    # Start the main conversation loop
    chat(agent)
