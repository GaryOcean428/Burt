import pkg_resources

# List of packages in your requirements.txt
required_packages = [
    'arrow==1.3.0', 'aiohttp==3.10.5', 'anthropic==0.34.1', 'anyio==4.4.0', 'backoff==2.2.1',
    'bcrypt==4.2.0', 'cachetools==5.5.0', 'certifi==2024.7.4', 'chardet==5.2.0',
    'chromadb==0.5.5', 'click==8.1.7', 'dataclasses-json==0.6.7', 'duckduckgo_search==6.1.12',
    'groq==0.10.0', 'httpx==0.27.2', 'huggingface-hub==0.24.6', 'langchain==0.2.15',
    'langchain-anthropic==0.1.23', 'langchain-chroma==0.1.2', 'langchain-community==0.2.9',
    'langchain-core==0.2.36', 'langchain-google-genai==1.0.7', 'langchain-groq==0.1.6',
    'langchain-huggingface==0.0.3', 'langchain-openai==0.1.15', 'numpy==1.26.4', 'openai==1.42.0',
    'pydantic==2.8.2', 'pydantic-settings==2.4.0', 'python-dotenv==1.0.1',
    'requests==2.32.3', 'tiktoken==0.7.0', 'tqdm==4.66.5', 'typing-extensions==4.12.2'
]

# Convert to a set for easier comparison
required_packages_set = {pkg.split("==")[0] for pkg in required_packages}

# List installed packages
installed_packages = {pkg.key for pkg in pkg_resources.working_set}

# Find the packages that are installed but not required
unnecessary_packages = installed_packages - required_packages_set

# Print the list of unnecessary packages
if unnecessary_packages:
    print("The following packages can be removed:")
    for pkg in sorted(unnecessary_packages):
        print(pkg)
else:
    print("No unnecessary packages found.")
