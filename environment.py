from command_executor import CommandExecutor
from state import *

class Environment:
    def __init__(self, repo_path: str):
        self.executor = CommandExecutor(timeout=100)
        self.repo_path = repo_path
        self.history = []

    def execute(self, action: Action) -> State:
        output = self.executor.execute(action, self.repo_path)
        state = State(action, output)
        self.history.append(state)
        return state