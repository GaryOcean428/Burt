from dataclasses import dataclass
import os, json, contextlib, subprocess, ast, shlex
from io import StringIO
import time
from typing import Literal
from app.python.helpers.tool import Tool, Response
from app.python.helpers import files, messages
from app.agent import Agent
from app.python.helpers.shell_local import LocalInteractiveSession
from app.python.helpers.shell_ssh import SSHSession as SSHInteractiveSession
from app.python.helpers.docker import DockerContainerManager
from app.python.helpers.print_style import PrintStyle


@dataclass
class State:
    shell: LocalInteractiveSession | SSHInteractiveSession
    docker: DockerContainerManager | None


class CodeExecution(Tool):

    def before_execution(self, runtime="", code="", **kwargs):
        # Validate the runtime argument
        valid_runtimes = ["python", "nodejs", "terminal", "output"]
        if runtime.lower().strip() not in valid_runtimes:
            raise ValueError(
                f"Invalid runtime. Must be one of: {', '.join(valid_runtimes)}"
            )

        # Validate the code argument
        if runtime != "output" and not code:
            raise ValueError("Code must be provided for execution")

    def execute(self, runtime="", code="", **kwargs):
        if self.agent.handle_intervention():
            return Response(
                message="", break_loop=False
            )  # wait for intervention and handle it, if paused

        self.prepare_state()

        runtime = runtime.lower().strip()
        if runtime == "python":
            response = self.execute_python_code(code)
        elif runtime == "nodejs":
            response = self.execute_nodejs_code(code)
        elif runtime == "terminal":
            response = self.execute_terminal_command(code)
        elif runtime == "output":
            response = self.get_terminal_output()
        else:
            response = files.read_file(
                "./prompts/fw.code_runtime_wrong.md", runtime=runtime
            )

        if not response:
            response = files.read_file("./prompts/fw.code_no_output.md")
        return Response(message=response, break_loop=False)

    def after_execution(self, response, **kwargs):
        msg_response = files.read_file(
            "./prompts/fw.tool_response.md",
            tool_name=self.name,
            tool_response=response.message,
        )
        self.agent.append_message(msg_response, human=True)

    def prepare_state(self):
        self.state = self.agent.get_data("cot_state")
        if not self.state:

            # initialize docker container if execution in docker is configured
            if self.agent.config["code_exec_docker_enabled"]:
                docker = DockerContainerManager(
                    name=self.agent.config["code_exec_docker_name"],
                    image=self.agent.config["code_exec_docker_image"],
                    ports=self.agent.config["code_exec_docker_ports"],
                    volumes=self.agent.config["code_exec_docker_volumes"],
                )
                docker.start_container()
            else:
                docker = None

            # initialize local or remote interactive shell interface
            if self.agent.config["code_exec_ssh_enabled"]:
                shell = SSHInteractiveSession(
                    self.agent.config["code_exec_ssh_addr"],
                    self.agent.config["code_exec_ssh_port"],
                    self.agent.config["code_exec_ssh_user"],
                    self.agent.config["code_exec_ssh_pass"],
                )
            else:
                shell = LocalInteractiveSession()

            self.state = State(shell=shell, docker=docker)
            shell.connect()
        self.agent.set_data("cot_state", self.state)

    def execute_python_code(self, code):
        escaped_code = shlex.quote(code)
        command = f"python3 -c {escaped_code}"
        return self.terminal_session(command)

    def execute_nodejs_code(self, code):
        escaped_code = shlex.quote(code)
        command = f"node -e {escaped_code}"
        return self.terminal_session(command)

    def execute_terminal_command(self, command):
        return self.terminal_session(command)

    def terminal_session(self, command):

        if self.agent.handle_intervention():
            return ""  # wait for intervention and handle it, if paused

        self.state.shell.execute_command(command)

        PrintStyle(background_color="white", font_color="#1B4F72", bold=True).print(
            f"{self.agent.agent_name} code execution output:"
        )
        return self.get_terminal_output()

    def get_terminal_output(self):
        idle = 0
        full_output = ""
        while True:
            time.sleep(0.1)  # Wait for some output to be generated
            output = self.state.shell.execute_command("")

            if self.agent.handle_intervention():
                return full_output  # wait for intervention and handle it, if paused

            if output:
                PrintStyle(font_color="#85C1E9").print(output)
                full_output += output
                idle = 0
            else:
                idle += 1
                if (full_output and idle > 30) or (not full_output and idle > 100):
                    return full_output

# Ensure the Tool class is available for import
Tool = CodeExecution
