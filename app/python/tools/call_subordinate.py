import sys
import os

# Add the project root to sys.path
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.append(project_root)

from app.agent import Agent
from app.python.helpers.tool import Tool, Response
from app.python.helpers import files
from app.python.helpers.print_style import PrintStyle


class Delegation(Tool):

    def execute(self, message="", reset="", **kwargs):
        # create subordinate agent using the data object on this agent and set superior agent to his data object
        if (
            self.agent.get_data("subordinate") is None
            or str(reset).lower().strip() == "true"
        ):
            subordinate = Agent(self.agent.number + 1, self.agent.config)
            subordinate.set_data("superior", self.agent)
            self.agent.set_data("subordinate", subordinate)
        # run subordinate agent message loop
        return Response(
            message=self.agent.get_data("subordinate").message_loop(message),
            break_loop=False,
        )
