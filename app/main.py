import os
import importlib
import sys
import inspect
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from app.advanced_router import AdvancedRouter
from app.agent import Agent, AgentConfig
from app.config import load_config
from app.python.helpers.tool import Tool
import logging
from dotenv import load_dotenv

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

# Add these lines near the top of the file, after the imports
from dotenv import load_dotenv
import os

load_dotenv()
perplexity_api_key = os.getenv("API_KEY_PERPLEXITY")
print(f"Perplexity API Key: {'Set' if perplexity_api_key else 'Not set'}")

# Load tools
def load_tools(agent):
    tools_dir = os.path.join(os.path.dirname(__file__), "python", "tools")
    tools = {
        obj(agent).name: obj(agent)
        for filename in os.listdir(tools_dir)
        if filename.endswith(".py") and filename != "__init__.py"
        for module_name in [f"app.python.tools.{filename[:-3]}"]
        for module in [importlib.import_module(module_name)]
        for _, obj in inspect.getmembers(module)
        if inspect.isclass(obj) and issubclass(obj, Tool) and obj != Tool
    }
    return list(tools.values())


# Load configuration
config = load_config()

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Set the template folder path
template_dir = os.path.join(project_root, "app", "templates")
static_dir = os.path.join(project_root, "app", "static")

app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)
CORS(app)

# Prepare the configuration for AgentConfig
agent_config_dict = {
    "chat_model": config["chat_model"],
    "utility_model": config["utility_model"],
    "backup_utility_model": config["backup_utility_model"],
    "embeddings_model": config["embeddings_model"],
    **{
        key: config.get(key, default)
        for key, default in [
            ("memory_subdir", ""),
            ("auto_memory_count", 3),
            ("auto_memory_skip", 2),
            ("rate_limit_seconds", 60),
            ("rate_limit_requests", 120),
            ("rate_limit_input_tokens", 200000),
            ("rate_limit_output_tokens", 200000),
            ("msgs_keep_max", 25),
            ("msgs_keep_start", 5),
            ("msgs_keep_end", 10),
            ("response_timeout_seconds", 60),
            ("max_tool_response_length", 3000),
            ("code_exec_docker_enabled", True),
            ("code_exec_docker_name", "agent-zero-exe"),
            ("code_exec_docker_image", "frdel/agent-zero-exe:latest"),
            ("code_exec_docker_ports", {"22/tcp": 50022}),
            ("code_exec_docker_volumes", {}),
            ("code_exec_ssh_enabled", True),
            ("code_exec_ssh_addr", "localhost"),
            ("code_exec_ssh_port", 50022),
            ("code_exec_ssh_user", "root"),
            ("code_exec_ssh_pass", "toor"),
        ]
    },
}

# Add this line after creating the AgentConfig
agent_config_dict["PERPLEXITY_API_KEY"] = os.getenv("API_KEY_PERPLEXITY")

# Initialize the Agent with a number (1) and the AgentConfig
agent_config = AgentConfig(**agent_config_dict)
agent = Agent(1, agent_config)

# Load tools and set them for the agent
tools = load_tools(agent)
agent.set_tools({tool.name: tool for tool in tools})

# Set use_tools attribute if it exists in the Agent class
if hasattr(agent, "use_tools"):
    setattr(agent, "use_tools", True)
if hasattr(agent, "use_memory"):
    setattr(agent, "use_memory", True)

# Initialize AdvancedRouter with config and agent
router = AdvancedRouter(config, agent)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/query", methods=["POST"])
async def query():
    data = request.json
    if data is None:
        return jsonify({"error": "Invalid JSON data"}), 400
    user_input = data.get("query", "")

    logging.info(f"Processing advanced query: {user_input[:20]}...")

    try:
        result = await router.process(user_input, "chat", {})
        logging.info(f"Router process result: {result}")

        selected_model = result.get("model_used", "Unknown")
        response_content = result.get("content", "No response")
        task_type = result.get("task_type", "Unknown")
        task_complexity = result.get("task_complexity", "Unknown")

        logging.info(f"Selected model: {selected_model}")
        logging.info(f"Task type: {task_type}")
        logging.info(f"Task complexity: {task_complexity}")
        logging.info(f"Response content: {response_content[:100]}...")

        response_metadata = {
            "model_used": selected_model,
            "task_type": task_type,
            "task_complexity": task_complexity,
        }

        return jsonify(
            {
                "response": response_content,
                "metadata": response_metadata,
            }
        )
    except Exception as e:
        logging.error(f"Unexpected error in query processing: {str(e)}", exc_info=True)
        return (
            jsonify({"error": "An unexpected error occurred", "details": str(e)}),
            500,
        )


if __name__ == "__main__":
    print("Loading tools...")
    loaded_tools = load_tools(agent)
    print(f"Loaded tools: {[tool.name for tool in loaded_tools]}")
    agent.set_tools({tool.name: tool for tool in loaded_tools})
    print(f"Agent tools: {list(agent.get_tools().keys())}")
    app.run(debug=config.get("DEBUG", False))
