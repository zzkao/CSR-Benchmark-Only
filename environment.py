import subprocess
import uuid
import atexit
from state import *
from command_executor import CommandExecutor
import os

class Environment:
    """
    Manages a benchmark Docker environment for LLM evaluation.
    Spawns an isolated container, executes commands, and tracks history.
    """

    def __init__(self, repo_path: str, image_name="benchmark-image", timeout=900):
        self.container_name = f"benchmark_{uuid.uuid4().hex}"
        self.repo_path = repo_path
        self.history = []

        # Start the container
        subprocess.run([
            "docker", "run", "-dit",
            "--name", self.container_name,
            "-v", f"{repo_path}:/workspace",
            image_name, "/bin/bash"
        ], check=True)

        # Create executor tied to this container
        self.executor = CommandExecutor(container_name=self.container_name, timeout=timeout)

        # Ensure cleanup on interpreter exit
        atexit.register(self.close)

    def execute(self, action: Action) -> State:
        """Executes an action in the container and stores the resulting state."""
        output = self.executor.execute(action)
        state = State(action, output)
        self.history.append(state)
        return state

    def close(self):
        """Close executor and remove container."""

        # remove tmp files
        os.remove(f'./tmp/cmd_stdout{self.container_name}.txt')
        os.remove(f'./tmp/cmd_stderr{self.container_name}.txt') 
        os.remove(f'./tmp/cmd_exit{self.container_name}.txt')

        if hasattr(self, "executor"):
            self.executor.close()
        subprocess.run(["docker", "rm", "-f", self.container_name],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


if __name__ == "__main__":

    with Environment(repo_path=".", image_name="benchmark-image") as env:
        while True:
            command = input('Enter a command to execute (or "exit" to quit): ')
            if command.lower() == 'exit':
                break
            action = Action(command=command)
            result = env.execute(action)
            print(result)