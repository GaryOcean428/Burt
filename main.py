import sys
import os
import threading
import atexit
import signal
from agent import Agent, AgentConfig
from python.helpers import files
from python.tools.memory_tool import initialize as init_memory
from python.tools.helper_agent_tool import call_helper_agents

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
    "Groq": ["llama-3.1-70b-versatile", "llama-3.1-70b", "llama-3.1-70b-base"],
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
            choice = int(
                input(f"\nSelect a {model_type} model (1-{len(all_models) + 1}): ")
            )
            if 1 <= choice <= len(all_models):
                return all_models[choice - 1]
            elif choice == len(all_models) + 1:
                return input("Enter the name of your custom model: ")
            else:
                print("Invalid choice. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a number.")


def get_model_instance(model_name):
    # This function should be implemented
    # to return the appropriate model instance
    # based on the selected model_name. For now, we'll return a placeholder.
    return f"Model instance for {model_name}"


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
    embedding_llm = get_model_instance(embedding_model_name)

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

    # Initialize memory
    init_memory(agent)

    print("Initialization successful!")
    return agent  # Return the created agent instead of exiting


def save_memory(agent):
    if agent:
        print("Saving memory state...")
        # Implement the actual memory saving logic here


def signal_handler(signum, frame):
    # We're not using signum and frame, but they're required for the signal handler signature
    print("\nInterrupt received, saving memory state and exiting...")
    save_memory(agent)
    sys.exit(0)


def capture_keys():
    # Implement the key capture logic here
    pass


def chat(agent):
    print("\nStarting chat session. Type 'exit' to end the conversation.")
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == "exit":
            print("Ending chat session.")
            break

        # Process user input and get a response from the agent
        try:
            agent.message_loop(user_input)
        except Exception as e:
            print(f"Error: An exception occurred while processing your input. {str(e)}")


if __name__ == "__main__":
    print("Initializing framework...")

    # Start the key capture thread for user intervention during agent streaming
    threading.Thread(target=capture_keys, daemon=True).start()

    # Start the initialization and get the created agent
    agent = initialize()

    # Update the atexit function with the created agent
    atexit.register(save_memory, agent=agent)

    # Register signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)

    # Start the main conversation loop
    chat(agent)
