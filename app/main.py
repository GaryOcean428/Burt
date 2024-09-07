import os
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

from app.advanced_router import AdvancedRouter
from app.agent import Agent
from app.config import load_config

# Get the absolute path to the project root
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Set the template folder path
template_dir = os.path.join(project_root, "app", "templates")

app = Flask(__name__, template_folder=template_dir)
CORS(app)

router = AdvancedRouter()

# Load configuration
config = load_config()

# Initialize the Agent with a number (1) and the config
agent = Agent(1, config)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/query", methods=["POST"])
def query():
    try:
        user_input = request.json["query"]

        # Use the AdvancedRouter to select the appropriate model
        selected_model = router.select_model(user_input)

        # Adjust parameters based on the task
        params = router.adjust_parameters(selected_model, user_input)

        # Use the Agent to process the query
        response = agent.process(user_input, selected_model, params)

        return jsonify({"response": response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
