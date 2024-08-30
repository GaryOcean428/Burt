# Agent Zero

[![Join our Discord](https://img.shields.io/badge/Discord-Join%20our%20server-5865F2?style=for-the-badge&logo=discord&logoColor=white)](https://discord.gg/B8KZKNsPpj) [![Subscribe on YouTube](https://img.shields.io/badge/YouTube-Subscribe-red?style=for-the-badge&logo=youtube&logoColor=white)](https://www.youtube.com/@AgentZeroFW) [![Connect on LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/jan-tomasek/) [![Follow on X.com](https://img.shields.io/badge/X.com-Follow-1DA1F2?style=for-the-badge&logo=x&logoColor=white)](https://x.com/JanTomasekDev)

[![Intro Video](/docs/intro_vid.jpg)](https://www.youtube.com/watch?v=C9n8zFpaV3I)

**Personal and organic AI framework**

- Agent Zero is not a predefined agentic framework. It is designed to be dynamic, organically growing, and learning as you use it.
- Agent Zero is fully transparent, readable, comprehensible, customizable and interactive.
- Agent Zero uses the computer as a tool to accomplish its (your) tasks.

## Key concepts

(... content unchanged ...)

## Setup

### For Windows Users (Preferred Method)

1. **Install Python:**
   - Download and install the latest version of Python from [python.org](https://www.python.org/downloads/)
   - During installation, make sure to check "Add Python to PATH"

2. **Set up the project:**
   Open Command Prompt and run:

   ```cmd
   git clone https://github.com/yourusername/agent-zero.git
   cd agent-zero
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Set up environment variables:**
   - Copy the `example.env` file to `.env`:

     ```cmd
     copy example.env .env
     ```

   - Edit the `.env` file with your preferred text editor and add your API keys.

4. **Run Agent Zero:**

   ```cmd
   python main.py
   ```

### For WSL (Windows Subsystem for Linux) Users

1. **Install WSL:**
   - Follow the [official Microsoft guide](https://docs.microsoft.com/en-us/windows/wsl/install) to install WSL 2 with Ubuntu.

2. **Set up the project:**
   Open a WSL terminal and run:

   ```bash
   git clone https://github.com/yourusername/agent-zero.git
   cd agent-zero
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Set up environment variables:**
   - Copy the `example.env` file to `.env`:

     ```bash
     cp example.env .env
     ```

   - Edit the `.env` file with your preferred text editor and add your API keys.

4. **Run Agent Zero:**

   ```bash
   python main.py
   ```

### Additional Setup Steps

1. **Required API keys:**
   - At the moment, the only recommended API key is for <https://www.perplexity.ai/> API. Perplexity is used as a convenient web search tool and has not yet been replaced by an open-source alternative. If you do not have an API key for Perplexity, leave it empty in the .env file and Perplexity will not be used.
   - Chat models and embedding models can be executed locally via Ollama and HuggingFace or via API as well.

2. **Choose your chat, utility and embeddings model:**
   - In the `main.py` file, at the start of the `chat()` function, you can see how the chat model and embedding model are set.
   - You can choose between online models (OpenAI, Anthropic, Groq) or offline (Ollama, HuggingFace) for both.

3. **Run Docker (Optional):**
   - If you want to use the Docker container, install Docker Desktop and run it. The framework will handle the rest.

## Troubleshooting

If you encounter any issues:

1. Make sure you're running the correct Python version (3.8+) in your environment.
2. Verify that all dependencies are correctly installed.
3. Check that your API keys are correctly set in the `.env` file.
4. If using WSL, ensure you're running the script from the WSL terminal, not the Windows Command Prompt.

For any persistent issues, please open an issue on the GitHub repository or seek help on our Discord server.

## Run the program

- In your configured environment (Windows or WSL), navigate to the project directory and run:

  ```bash
  python main.py
  ```

- Or run it in debug mode in VS Code using the "debug" button in the top right corner of the editor. Config files for VS Code are provided for this purpose.

(... rest of the content unchanged ...)
