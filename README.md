# Burton

[![Join our Discord](https://img.shields.io/badge/Discord-Join%20our%20server-5865F2?style=for-the-badge&logo=discord&logoColor=white)](https://discord.gg/B8KZKNsPpj) [![Subscribe on YouTube](https://img.shields.io/badge/YouTube-Subscribe-red?style=for-the-badge&logo=youtube&logoColor=white)](https://www.youtube.com/@AgentZeroFW) [![Connect on LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/jan-tomasek/) [![Follow on X.com](https://img.shields.io/badge/X.com-Follow-1DA1F2?style=for-the-badge&logo=x&logoColor=white)](https://x.com/JanTomasekDev)

[![Intro Video](/docs/intro_vid.jpg)](https://www.youtube.com/watch?v=C9n8zFpaV3I)

## Personal and organic AI framework

- Burton is not a predefined agentic framework. It is designed to be dynamic, organically growing, and learning as you use it.
- Burton is fully transparent, readable, comprehensible, customizable and interactive.
- Burton uses the computer as a tool to accomplish its (your) tasks.

## Project Overview

Burton is a sophisticated AI agent framework designed to facilitate the creation and management of multiple agents with hierarchical relationships.

## Project Structure

The project is organized as follows:

- `app/agent.py`: Contains the `Agent` class and its configuration.
- `app/python/tools/call_subordinate.py`: Implements the `Delegation` tool for managing subordinate agents.
- `app/python/helpers`: Contains various helper modules.
- `app/tests/test_agent.py`: Contains the test suite for the Agent class.

## Setup Instructions

### Prerequisites

- Python 3.12
- Docker (for code execution in a Docker container)
- SSH (for remote code execution)

### Installation

1. Clone the repository:

    ```sh
    git clone https://github.com/GaryOcean428/Burt.git
    cd burton
    ```

2. Create a virtual environment and activate it:

    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the dependencies:

    ```sh
    pip install -r requirements.txt
    ```

4. Ensure Docker is running and properly configured.

### Configuration

Update the `AgentConfig` in `app/agent.py` as needed. Here is an example configuration:

```tree

burton/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── advanced_router.py
│   ├── agent.py
│   ├── config.py
│   ├── models.py
│   └── python/
│       ���── helpers/
│       │   ├── __init__.py
│       │   ├── extract_tools.py
│       │   ├── errors.py
│       │   └── ... (other helper modules)
│       └── tools/
│           ├── __init__.py
│           ├── knowledge_tool.py
│           └── ... (other tool modules)
├── docs/
├── prompts/
├── templates/
├── .env
├── requirements.txt
└── README.md
```

## Setup

### For All Users (Recommended Method)

1. **Install Python:**
   - Download and install Python 3.12 or later from [python.org](https://www.python.org/downloads/)
   - During installation, make sure to check "Add Python to PATH"

2. **Install Poetry:**
   - Follow the installation instructions for Poetry from the [official documentation](https://python-poetry.org/docs/#installation)

3. **Set up the project:**
   Open a terminal (Command Prompt on Windows, Terminal on macOS/Linux) and run:

   ```bash
   git clone https://github.com/yourusername/burton.git
   cd burton
   poetry install
   ```

4. **Activate the virtual environment:**

   ```bash
   poetry shell
   ```

5. **Set up environment variables:**
   - Copy the `example.env` file to `.env`:

     ```bash
     cp example.env .env  # On Windows, use: copy example.env .env
     ```

   - Open the `.env` file and fill in your API keys and other configuration details

6. **Run the application:**

   ```bash
   poetry run python app/main.py
   ```

### For Windows Users (Alternative Method)

If you prefer using Anaconda, follow these steps:

1. **Install Anaconda:**
   - Download and install Anaconda from [anaconda.com](https://www.anaconda.com/products/distribution)

2. **Set up the project:**
   Open Anaconda Prompt and run:

   ```cmd
   git clone https://github.com/yourusername/burton.git
   cd burton
   conda env create -f environment.yml
   conda activate burton_env
   ```

3. **Set up environment variables:**
   - Copy the `example.env` file to `.env`:

     ```cmd
     copy example.env .env
     ```

   - Open the `.env` file and fill in your API keys and other configuration details

4. **Run the application:**

   ```cmd
   python app/main.py
   ```

## Development

To add new dependencies to the project:

```bash
poetry add package_name
```

To update dependencies:

```bash
poetry update
```

To run tests (if available):

```bash
poetry run pytest
```

## Troubleshooting

### Rate Limiting

If you encounter a "rate limit exceeded" error, the application will automatically retry the request up to 3 times. If the issue persists, please wait a few minutes before trying again or check your OpenAI API usage limits.

### AI Models

This project uses Claude 3.5 Sonnet as the default chat model, Groq's LLaMA 3.1 8B as the utility model, and GPT-4o-mini as the backup utility model. Make sure your API keys have access to these models. You can configure different models by setting the CHAT_MODEL, UTILITY_MODEL, and BACKUP_UTILITY_MODEL environment variables in your .env file.

If problems persist, please open an issue on the GitHub repository with details about the error and your environment.

## Run the program

- In your configured environment (Windows or WSL), navigate to the project directory and run:

  ```bash
  python app/main.py
  ```

- Or run it in debug mode in VS Code using the "debug" button in the top right corner of the editor. Config files for VS Code are provided for this purpose.

## Running Tests

To run the test suite and verify the project's functionality:

1. Make sure you're in the project root directory and your virtual environment is activated.

2. Run the following command:

   ```bash
   python -m unittest discover app/tests
   ```

   This command will discover and run all tests in the `app/tests` directory.

3. Review the test output to ensure all tests pass successfully.

## Environment Variables Setup

1. Copy the `.devcontainer/.env.example` file to `.devcontainer/.env`:

   ```sh
   cp .devcontainer/.env.example .devcontainer/.env
   ```

2. Open the `.devcontainer/.env` file in a text editor.

3. Replace all placeholder values with your actual API keys and credentials. Do not share this file or commit it to version control.

4. Save the file and close the editor.

IMPORTANT: Never commit your `.env` file to version control. It contains sensitive information that should be kept private.
