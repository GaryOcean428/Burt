import os
import importlib
import sys
import inspect
from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
from app.advanced_router import AdvancedRouter
from app.agent import Agent, AgentConfig
from app.config import load_config
from app.python.helpers.tool import Tool
from app.python.helpers.rag_system import RAGSystem
import logging
from dotenv import load_dotenv
import uuid
from app.python.helpers.redis_cache import RedisCache
import PyPDF2
import docx
import io
import nltk

# Download NLTK resources
nltk.download("punkt_tab", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load and check environment variables
perplexity_api_key = os.getenv("PERPLEXITY_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_environment = os.getenv("PINECONE_ENVIRONMENT")
pinecone_index_name = os.getenv("PINECONE_INDEX_NAME")

logger.info(
    f"Perplexity API Key: {'Set' if perplexity_api_key else 'Not set'}"
)
logger.info(f"Pinecone API Key: {'Set' if pinecone_api_key else 'Not set'}")
logger.info(f"Pinecone Environment: {pinecone_environment}")
logger.info(f"Pinecone Index Name: {pinecone_index_name}")

if not perplexity_api_key:
    logger.warning(
        "Perplexity API key is not set. Some features may not work properly."
    )

if not pinecone_api_key or not pinecone_environment or not pinecone_index_name:
    logger.warning(
        "One or more Pinecone configuration variables are not set. RAG system may not work properly."
    )


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

# Set the template folder path
template_dir = os.path.join(project_root, "app", "templates")
static_dir = os.path.join(project_root, "app", "static")

app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)
CORS(app)

# Set up file upload configuration
UPLOAD_FOLDER = os.path.join(project_root, "uploads")
ALLOWED_EXTENSIONS = {"pdf", "txt", "docx"}
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Prepare the configuration for AgentConfig
agent_config_dict = {
    "chat_model": config["chat_model"],
    "utility_model": config["utility_model"],
    "backup_utility_model": config["backup_utility_model"],
    "embeddings_model": config["embeddings_model"],
    "perplexity_api_key": perplexity_api_key,
    "pinecone_api_key": pinecone_api_key,
    "pinecone_environment": pinecone_environment,
    "pinecone_index_name": pinecone_index_name,
    "pinecone_dimension": int(os.getenv("PINECONE_DIMENSION", 1536)),
    "pinecone_cloud": os.getenv("PINECONE_CLOUD", "aws"),
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

# Load tools and set them for the agent
tools = load_tools(agent)
agent.set_tools({tool.name: tool for tool in tools})

# Set use_tools attribute if it exists in the Agent class
if hasattr(agent, "use_tools"):
    setattr(agent, "use_tools", True)
if hasattr(agent, "use_memory"):
    setattr(agent, "use_memory", True)

# Initialize RAGSystem
rag_system = RAGSystem()

# Initialize AdvancedRouter with config, agent, and RAGSystem
router = AdvancedRouter(config, agent, rag_system)


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def extract_text_from_file(file_path):
    file_extension = os.path.splitext(file_path)[1].lower()

    if file_extension == ".pdf":
        with open(file_path, "rb") as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
    elif file_extension == ".txt":
        with open(file_path, "r", encoding="utf-8") as txt_file:
            text = txt_file.read()
    elif file_extension == ".docx":
        doc = docx.Document(file_path)
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
    else:
        raise ValueError(f"Unsupported file type: {file_extension}")

    return text


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        try:
            text = extract_text_from_file(filepath)
            rag_system.add_document(text, metadata={"filename": filename})

            return (
                jsonify(
                    {
                        "message": "File uploaded and processed successfully",
                        "filename": filename,
                    }
                ),
                200,
            )
        except Exception as e:
            logger.error(f"Error processing file: {str(e)}")
            return jsonify({"error": f"Error processing file: {str(e)}"}), 500
    else:
        return jsonify({"error": "File type not allowed"}), 400


@app.route("/query", methods=["POST"])
async def query():
    data = request.json
    if data is None:
        return jsonify({"error": "Invalid JSON data"}), 400
    user_input = data.get("query", "")
    conversation_id = data.get("conversation_id", str(uuid.uuid4()))

    logger.info(f"Processing advanced query: {user_input[:20]}...")

    try:
        conversation_history = (
            RedisCache.get(f"conversation:{conversation_id}") or []
        )
    except Exception as e:
        logger.error(
            f"Error retrieving conversation history from Redis: {str(e)}"
        )
        conversation_history = []

    conversation_history.append({"role": "user", "content": user_input})

    try:
        result = await router.process(
            user_input, "chat", {"conversation_history": conversation_history}
        )
        logger.info(f"Router process result: {result}")

        selected_model = result.get("model_used", "Unknown")
        response_content = result.get("content", "No response")
        task_type = result.get("task_type", "Unknown")
        task_complexity = result.get("task_complexity", "Unknown")

        logger.info(f"Selected model: {selected_model}")
        logger.info(f"Task type: {task_type}")
        logger.info(f"Task complexity: {task_complexity}")
        logger.info(f"Response content: {response_content[:100]}...")

        conversation_history.append(
            {"role": "assistant", "content": response_content}
        )

        try:
            RedisCache.set(
                f"conversation:{conversation_id}", conversation_history
            )
        except Exception as e:
            logger.error(
                f"Error saving conversation history to Redis: {str(e)}"
            )

        response_metadata = {
            "model_used": selected_model,
            "task_type": task_type,
            "task_complexity": task_complexity,
            "conversation_id": conversation_id,
        }

        return jsonify(
            {
                "response": response_content,
                "metadata": response_metadata,
            }
        )
    except Exception as e:
        logger.error(
            f"Unexpected error in query processing: {str(e)}", exc_info=True
        )
        return (
            jsonify(
                {"error": "An unexpected error occurred", "details": str(e)}
            ),
            500,
        )


if __name__ == "__main__":
    logger.info("Loading tools...")
    loaded_tools = load_tools(agent)
    logger.info(f"Loaded tools: {[tool.name for tool in loaded_tools]}")
    agent.set_tools({tool.name: tool for tool in loaded_tools})
    logger.info(f"Agent tools: {list(agent.get_tools().keys())}")
    logger.info(f"RAG system initialized: {rag_system is not None}")
    app.run(debug=config.get("DEBUG", False))
