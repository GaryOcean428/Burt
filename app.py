from dotenv import load_dotenv
load_dotenv()

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from advanced_router import AdvancedRouter
from agent import Agent, AgentConfig
from models import get_openai_chat, get_openai_embedding  # Assuming these functions exist in models.py

app = Flask(__name__)
CORS(app)

router = AdvancedRouter()

# Create an AgentConfig object
config = AgentConfig(
    chat_model=get_openai_chat("gpt-3.5-turbo"),
    utility_model=get_openai_chat("gpt-3.5-turbo"),
    embeddings_model=get_openai_embedding("text-embedding-ada-002")
)

# Initialize the Agent with a number (1) and the config
agent = Agent(1, config)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/query', methods=['POST'])
def query():
    try:
        user_input = request.json['query']
        response = agent.message_loop(user_input)  # Changed from run to message_loop
        return jsonify({'response': response})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
