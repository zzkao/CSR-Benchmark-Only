import subprocess
import uuid
import atexit
from state import *
from command_executor import CommandExecutor
from script_evaluator import ScriptEvaluator
import os
import time
import json
from datetime import datetime
from collections import defaultdict

class Environment:
    """
    Manages a benchmark Docker environment for LLM evaluation.
    Spawns an isolated container, executes commands, and tracks history.
    """

    def __init__(self, repo_url: str, keep_repo=False, keep_docker=False, image_name="benchmark-image", timeout=900, verbose=False):
        self.verbose = verbose
        self.keep_docker = keep_docker
        self.keep_repo = keep_repo # (TODO) UNUSED FOR NOW
        
        # Docker Container
        self.container_name = f"benchmark_{uuid.uuid4().hex}"
        print(f"Agent running in container {self.container_name}")

        # Define repo path
        self.REPO_NAME= repo_url.rsplit('/', 1)[-1]
        self.name = f"{self.REPO_NAME}_{self.container_name}"
        self.repo_path = os.path.abspath(f"./data/CSRBench100/{self.name}/")

        # Pull the repo
        subprocess.run(
            ["git", "clone", repo_url, f"{self.repo_path}"],
            check=True
        )
        print(f"Repo cloned successfully to {self.repo_path}")

        print(f"Agent environment root at {self.repo_path}")

        # Histories
        self.history = defaultdict(list)

        # Start the container
        subprocess.run([
            "docker", "run", "-dit",
            "--name", self.container_name,
            "-v", f"{self.repo_path}:/workspace",
            image_name, "/bin/bash"
        ], check=True)

        # Create executor tied to this container
        self.executor = CommandExecutor(container_name=self.container_name, timeout=timeout)
        self.evaluator = ScriptEvaluator()

        # Ensure cleanup on interpreter exit
        atexit.register(self.close)

    def execute(self, action: Action) -> State:
        """Executes an action in the container and stores the resulting state."""
        output = self.executor.execute(action)
        state = State(action, output)
        self.history[action.agent_name].append(state)
        if self.verbose:
            print(state)
        return state
    
    def close(self):
        """Close executor and remove container."""
        self.log_environment_history()

        if hasattr(self, "executor"):
            self.executor.close()

        if not self.keep_docker:
            subprocess.run(["docker", "rm", "-f", self.container_name],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    def _read_test_script_commands(self, bash_file):
        commands = []
        current_cmd = []

        with open(bash_file) as f:
            for line in f:
                line = line.strip()
                # skip comments and empty lines
                if not line or line.startswith("#"):
                    continue

                # if line ends with \, continue building the command
                if line.endswith("\\"):
                    current_cmd.append(line[:-1].strip())
                else:
                    current_cmd.append(line)
                    # end of a full command
                    commands.append(" ".join(current_cmd))
                    current_cmd = []

        return commands

    def run_test_scripts(self, num):
        test_filepath = f"./data/CSR_bench_scripts/{num}.sh"
        description = f"TEST SCRIPT COMMAND"
        print(f"RUNNING TEST SCRIPT")

        # Return to base directory
        self.execute(Action("cd /workspace", agent_name="TEST"))
        self.execute(Action("ls", agent_name="TEST"))

        commands = self._read_test_script_commands(test_filepath)
        success = 0
        for command in commands:
            test_script_command = Action(f"{command}", description=description, agent_name="TEST")
            state = self.execute(test_script_command)
            output = state.output

            result = self.evaluator.query(bash_script=command, output=output)
            if result:
                success += 1
                self.history['TEST'][-1].set_eval('SUCCESS')
            else:
                self.history['TEST'][-1].set_eval('FAILED')
        
        total = len(commands)
        return f"{success} / {total}"

    def log_environment_history(self, pretty=True):
        for agent_name in self.history.keys():

            log_file = f"logs/{self.name}_{agent_name}.jsonl"

            # Ensure directory exists
            os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)

            with open(log_file, "a", encoding="utf-8") as f:
                for state in self.history[agent_name]:
                    log_entry = {
                        "timestamp": datetime.utcnow().isoformat() + "Z",
                        "state": state.to_dict() if not pretty else str(state)
                    }
                    if pretty:
                        f.write(f"{log_entry['timestamp']}\n{log_entry['state']}\n{'-'*60}\n")
                    else:
                        f.write(json.dumps(log_entry) + "\n")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


# Example usage
if __name__ == "__main__":
    with Environment(repo_url="https://github.com/stanford-oval/storm", image_name="benchmark-image") as env:

        while True:
            command = input('Enter a command to execute (or "exit" to quit): ')
            if command.lower() == 'exit':
                break
            action = Action(command=command)
            result = env.execute(action)
            print(result)
        print(env.history)
