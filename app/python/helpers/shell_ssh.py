import paramiko


class SSHSession:
    def __init__(self, hostname, port, username, password):
        self.hostname = hostname
        self.port = port
        self.username = username
        self.password = password
        self.client = None

    def connect(self):
        try:
            self.client = paramiko.SSHClient()
            self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            self.client.connect(
                hostname=self.hostname,
                port=self.port,
                username=self.username,
                password=self.password,
            )
            return True
        except Exception as e:
            print(f"Error connecting to SSH: {str(e)}")
            return False

    def execute_command(self, command):
        if not self.client:
            return "Error: Not connected to SSH"

        try:
            stdin, stdout, stderr = self.client.exec_command(command)
            return stdout.read().decode("utf-8")
        except Exception as e:
            return f"Error executing command: {str(e)}"

    def close(self):
        if self.client:
            self.client.close()
            self.client = None
