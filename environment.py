import subprocess
import uuid
import atexit
from state import *
from command_executor import CommandExecutor
import os
import time

class Environment:
    """
    Manages a benchmark Docker environment for LLM evaluation.
    Spawns an isolated container, executes commands, and tracks history.
    """

    def __init__(self, repo_path: str, image_name="benchmark-image", timeout=900):
        self.container_name = f"benchmark_{uuid.uuid4().hex}"
        self.repo_path = os.path.abspath(repo_path)
        self.history = []

        # Temp file names
        self.tmp = f'{self.repo_path}/tmp'
        self.stdout_file = f'{self.repo_path}/tmp/cmd_stdout{self.container_name}.txt'
        self.stderr_file = f'{self.repo_path}/tmp/cmd_stderr{self.container_name}.txt'
        self.exit_code_file = f'{self.repo_path}/tmp/cmd_exit{self.container_name}.txt'

        # Make temp files to track terminal outputs
        os.makedirs(self.tmp, exist_ok=True)
        open(self.stdout_file, 'w').close()
        open(self.stderr_file, 'w').close()
        open(self.exit_code_file, 'w').close()

        # Start the container
        subprocess.run([
            "docker", "run", "-dit",
            "--name", self.container_name,
            "-v", f"{self.repo_path}:/workspace",
            image_name, "/bin/bash"
        ], check=True)

        # Create executor tied to this container
        self.executor = CommandExecutor(container_name=self.container_name, timeout=timeout)

        # Ensure cleanup on interpreter exit
        atexit.register(self.close)

    def execute(self, action: Action) -> State:
        """Executes an action in the container and stores the resulting state."""
        if self.executor.execute(action) == 1:
            time.sleep(1)
            stdout, stderr, exit_code = self.read_std()
            output = BashOutput(stdout=stdout, stderr=stderr, exit_code=exit_code)
        state = State(action, output)
        self.history.append(state)
        return state

    def read_std(self):
            with open(self.stdout_file, 'r') as file:
                stdout_output = file.read().strip()
            with open(self.stderr_file, 'r') as file:
                stderr_output = file.read().strip()
            with open(self.exit_code_file, 'r') as file:
                exit_code = file.read().strip()
            
            return stdout_output.replace("\x00", ""), stderr_output.replace("\x00", ""), exit_code.replace("\x00", "")
    
    def close(self):
        """Close executor and remove container."""

        # remove tmp files
        os.remove(f'{self.stderr_file}')
        os.remove(f'{self.stdout_file}') 
        os.remove(f'{self.exit_code_file}')
        os.removedirs(f"{self.tmp}")

        if hasattr(self, "executor"):
            self.executor.close()
        subprocess.run(["docker", "rm", "-f", self.container_name],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


# Example usage
if __name__ == "__main__":
    with Environment(repo_path="data/CSRBench100/storm", image_name="benchmark-image") as env:
        while True:
            command = input('Enter a command to execute (or "exit" to quit): ')
            if command.lower() == 'exit':
                break
            action = Action(command=command)
            result = env.execute(action)
            print(result)
        print(env.history)