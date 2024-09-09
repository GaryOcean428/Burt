import os
import importlib
import sys
import inspect
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from app.advanced_router import AdvancedRouter
from app.agent import Agent, AgentConfig
from app.config import load_config
from app.models import get_chat_model, get_embedding_model
from app.python.helpers.tool import Tool
import logging

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# Load tools
def load_tools(agent):
    tools = {}
    tools_dir = os.path.join(os.path.dirname(__file__), 'python', 'tools')
    for filename in os.listdir(tools_dir):
        if filename.endswith('.py') and filename != '__init__.py':
            module_name = f'app.python.tools.{filename[:-3]}'
            try:
                module = importlib.import_module(module_name)
                print(f"Loaded module: {module_name}")
                for name, obj in inspect.getmembers(module):
                    if inspect.isclass(obj) and issubclass(obj, Tool) and obj != Tool:
                        print(f"Found tool class: {name}")
                        tool_instance = obj(agent)
                        if tool_instance.name not in tools:
                            tools[tool_instance.name] = tool_instance
                            print(f"Added tool: {tool_instance.name}")
                        else:
                            print(f"Skipped duplicate tool: {tool_instance.name}")
            except Exception as e:
                print(f"Error loading tool from {filename}: {str(e)}")
    return list(tools.values())

# Load configuration
config = load_config()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Get the absolute path to the project root
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Set the template folder path
template_dir = os.path.join(project_root, "app", "templates")

app = Flask(__name__, template_folder=template_dir)
CORS(app)

router = AdvancedRouter(config)

# Prepare the configuration for AgentConfig
agent_config_dict = {
    "chat_model": get_chat_model(config["chat_model"]),
    "utility_model": get_chat_model(config["utility_model"]),
    "backup_utility_model": get_chat_model(config["backup_utility_model"]),
    "embeddings_model": get_embedding_model(config["embeddings_model"]),
    "memory_subdir": config.get("memory_subdir", ""),
    "auto_memory_count": config.get("auto_memory_count", 3),
    "auto_memory_skip": config.get("auto_memory_skip", 2),
    "rate_limit_seconds": config.get("rate_limit_seconds", 60),
    "rate_limit_requests": config.get("rate_limit_requests", 120),
    "rate_limit_input_tokens": config.get("rate_limit_input_tokens", 200000),
    "rate_limit_output_tokens": config.get("rate_limit_output_tokens", 200000),
    "msgs_keep_max": config.get("msgs_keep_max", 25),
    "msgs_keep_start": config.get("msgs_keep_start", 5),
    "msgs_keep_end": config.get("msgs_keep_end", 10),
    "response_timeout_seconds": config.get("response_timeout_seconds", 60),
    "max_tool_response_length": config.get("max_tool_response_length", 3000),
    "code_exec_docker_enabled": config.get("code_exec_docker_enabled", True),
    "code_exec_docker_name": config.get("code_exec_docker_name", "agent-zero-exe"),
    "code_exec_docker_image": config.get("code_exec_docker_image", "frdel/agent-zero-exe:latest"),
    "code_exec_docker_ports": config.get("code_exec_docker_ports", {"22/tcp": 50022}),
    "code_exec_docker_volumes": config.get("code_exec_docker_volumes", {}),
    "code_exec_ssh_enabled": config.get("code_exec_ssh_enabled", True),
    "code_exec_ssh_addr": config.get("code_exec_ssh_addr", "localhost"),
    "code_exec_ssh_port": config.get("code_exec_ssh_port", 50022),
    "code_exec_ssh_user": config.get("code_exec_ssh_user", "root"),
    "code_exec_ssh_pass": config.get("code_exec_ssh_pass", "toor"),
}

# Initialize the Agent with a number (1) and the AgentConfig
agent_config = AgentConfig(**agent_config_dict)
agent = Agent(1, agent_config)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/query", methods=["POST"])
def query():
    try:
        user_input = request.json["query"]

        if not user_input.strip():
            return jsonify({"error": "Empty input. Please provide a valid query."}), 400

        # Use the AdvancedRouter to select the appropriate model
        selected_model = router.select_model(user_input)

        # Adjust parameters based on the task
        params = router.adjust_parameters(selected_model, user_input)

        # Use the Agent to process the query
        response = agent.process(user_input, selected_model, params)

        return jsonify({"response": response})
    except KeyError:
        logging.error("KeyError: 'query' not found in request JSON")
        return jsonify({"error": "Invalid request. 'query' field is missing."}), 400
    except Exception as e:
        logging.error(f"Error processing query: {str(e)}", exc_info=True)
        return jsonify({"error": f"An error occurred while processing the request: {str(e)}"}), 500

if __name__ == "__main__":
    print("Loading tools...")
    loaded_tools = load_tools(agent)
    print(f"Loaded tools: {[tool.name for tool in loaded_tools]}")
    agent.set_tools({tool.name: tool for tool in loaded_tools})
    print(f"Agent tools: {list(agent.get_tools().keys())}")
    app.run(debug=config.get("DEBUG", False))
