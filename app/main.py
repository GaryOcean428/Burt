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

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)


# Load tools
def load_tools(agent):
    tools_dir = os.path.join(os.path.dirname(__file__), "python", "tools")
    tools = {
        obj(agent).name: obj(agent)
        for filename in os.listdir(tools_dir)
        if filename.endswith(".py") and filename != "__init__.py"
        for module_name in [f"app.python.tools.{filename[:-3]}"]
        for module in [importlib.import_module(module_name)]
        for name, obj in inspect.getmembers(module)
        if inspect.isclass(obj) and issubclass(obj, Tool) and obj != Tool
    }
    return list(tools.values())


# Load configuration
config = load_config()

# Set up logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Set the template folder path
template_dir = os.path.join(project_root, "app", "templates")

app = Flask(__name__, template_folder=template_dir)
CORS(app)

router = AdvancedRouter(config)

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

# Initialize the Agent with a number (1) and the AgentConfig
agent_config = AgentConfig(**agent_config_dict)
agent = Agent(1, agent_config)


# Add an additional blank line
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/query", methods=["POST"])
def query():
    data = request.json
    user_input = data.get("query", "")

    logging.info(f"Processing advanced query: {user_input[:20]}...")

    try:
        result = router.process_query_advanced(user_input)
        selected_model = result["model"]
        task_type = result["task_type"]
        task_complexity = result["task_complexity"]

        logging.info(f"Selected model: {selected_model}")
        logging.info(f"Task type: {task_type}")
        logging.info(f"Task complexity: {task_complexity}")

        params = router.get_model_parameters(task_type, task_complexity)
        logging.info(f"Model parameters: {params}")

        agent = Agent(1, agent_config)
        agent.set_tools({tool.name: tool for tool in load_tools(agent)})

        response = agent.process(user_input, selected_model, params)

        return jsonify(
            {
                "response": response,
                "model_used": selected_model,
                "task_type": task_type,
                "task_complexity": task_complexity,
            }
        )
    except Exception as e:
        logging.error(f"Error processing query: {str(e)}")
        return (
            jsonify(
                {
                    "error": "An error occurred while processing your request. Please try again.",
                    "details": str(e),
                }
            ),
            500,
        )


if __name__ == "__main__":
    print("Loading tools...")
    loaded_tools = load_tools(agent)
    print(f"Loaded tools: {[tool.name for tool in loaded_tools]}")
    agent.set_tools({tool.name: tool for tool in loaded_tools})
    print(f"Agent tools: {list(agent.get_tools().keys())}")
    app.run(debug=config.get("DEBUG", False))
