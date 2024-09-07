from typing import Dict, Any
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Configuration constants
ROUTER_THRESHOLD = 0.7


def load_config() -> Dict[str, Any]:
    return {
        "ROUTER_THRESHOLD": ROUTER_THRESHOLD,
        "rate_limit_requests": int(os.getenv("RATE_LIMIT_REQUESTS", 120)),
        "rate_limit_seconds": int(os.getenv("RATE_LIMIT_SECONDS", 60)),
        "rate_limit_input_tokens": int(os.getenv("RATE_LIMIT_INPUT_TOKENS", 200000)),
        "rate_limit_output_tokens": int(os.getenv("RATE_LIMIT_OUTPUT_TOKENS", 200000)),
        "msgs_keep_max": int(os.getenv("MSGS_KEEP_MAX", 25)),
        "msgs_keep_start": int(os.getenv("MSGS_KEEP_START", 5)),
        "msgs_keep_end": int(os.getenv("MSGS_KEEP_END", 10)),
        "response_timeout_seconds": int(os.getenv("RESPONSE_TIMEOUT_SECONDS", 60)),
        "max_tool_response_length": int(os.getenv("MAX_TOOL_RESPONSE_LENGTH", 3000)),
        "memory_subdir": os.getenv("MEMORY_SUBDIR", ""),
        "auto_memory_count": int(os.getenv("AUTO_MEMORY_COUNT", 3)),
        "auto_memory_skip": int(os.getenv("AUTO_MEMORY_SKIP", 2)),
        "code_exec_docker_enabled": os.getenv(
            "CODE_EXEC_DOCKER_ENABLED", "True"
        ).lower()
        == "true",
        "code_exec_docker_name": os.getenv("CODE_EXEC_DOCKER_NAME", "agent-zero-exe"),
        "code_exec_docker_image": os.getenv(
            "CODE_EXEC_DOCKER_IMAGE", "frdel/agent-zero-exe:latest"
        ),
        "code_exec_docker_ports": {
            "22/tcp": int(os.getenv("CODE_EXEC_DOCKER_PORT", 50022))
        },
        "code_exec_docker_volumes": {
            os.path.abspath(os.getenv("WORK_DIR", "work_dir")): {
                "bind": "/root",
                "mode": "rw",
            }
        },
        "code_exec_ssh_enabled": os.getenv("CODE_EXEC_SSH_ENABLED", "True").lower()
        == "true",
        "code_exec_ssh_addr": os.getenv("CODE_EXEC_SSH_ADDR", "localhost"),
        "code_exec_ssh_port": int(os.getenv("CODE_EXEC_SSH_PORT", 50022)),
        "code_exec_ssh_user": os.getenv("CODE_EXEC_SSH_USER", "root"),
        "code_exec_ssh_pass": os.getenv("CODE_EXEC_SSH_PASS", ""),
        "chat_model": os.getenv("CHAT_MODEL", "claude-3.5-sonnet"),
        "utility_model": os.getenv("UTILITY_MODEL", "llama-3.1-8b"),
        "backup_utility_model": os.getenv("BACKUP_UTILITY_MODEL", "gpt-4o-mini"),
        "embeddings_model": os.getenv("EMBEDDINGS_MODEL", "text-embedding-ada-002"),
    }


# Add any other configuration variables as needed
