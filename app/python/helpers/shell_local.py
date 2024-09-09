import subprocess
import shlex


def execute_command(command):
    try:
        args = shlex.split(command)
        result = subprocess.run(args, capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        return f"Error: {e.stderr}"
    except Exception as e:
        return f"Error: {str(e)}"


class LocalInteractiveSession:
    def __init__(self):
        self.process = None

    def start(self, command):
        try:
            self.process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                shell=True,
            )
        except Exception as e:
            return f"Error starting session: {str(e)}"

    def execute(self, command):
        if not self.process:
            return "Error: Session not started"

        try:
            self.process.stdin.write(command + "\n")
            self.process.stdin.flush()
            return self.process.stdout.readline().strip()
        except Exception as e:
            return f"Error executing command: {str(e)}"

    def close(self):
        if self.process:
            self.process.terminate()
            self.process = None
