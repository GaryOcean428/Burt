from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

from .advanced_router import AdvancedRouter
from .agent import Agent
from .config import load_config

app = Flask(__name__)
CORS(app)

router = AdvancedRouter()

# Load configuration
config = load_config()

# Initialize the Agent with a number (1) and the config
agent = Agent(1, config)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/query', methods=['POST'])
def query():
    try:
        user_input = request.json['query']
        response = agent.message_loop(user_input)
        return jsonify({'response': response})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
