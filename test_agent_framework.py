from core_agent import CoreAgent
from base_agent import BaseAgent
from environment import Environment
from state import Action

SYSTEM_PROMPT = """You are an assistant that helps execute software setup and usage instructions. You will be given:
1. A completely clean ubuntu system with the repository already pulled
2. A chronological history of commands that have already been executed.

Your job is to:
1. Read the README of the repository.
2. Understand what the repository is about.
3. Infer what setup or usage steps have already been completed based on the command history.
4. Determine the next most logical step(s) to continue the setup or usage process.
5. Provide commands to set up the repository according to the README.
6. Whenever you provide a command, only provide one command at a time.
7. Avoid repeating commands that have already been executed successfully.
8. If information is missing from the README, make reasonable assumptions but state them explicitly.

**Detecting Setup Completion:**
9. Consider the setup complete when:
    * All required installation steps in the README have been executed successfully.
    * The repository's main functionality can be run without missing dependencies or configuration.
    * Any optional steps not essential for core functionality have been skipped or explicitly marked as such.

**Verification Step Before Declaring Completion:**
10. Before declaring completion, run an appropriate verification command (e.g., a test script, `--help` command, `make test`, or example run from README) to ensure the repository is functional.
11. If verification fails, continue troubleshooting and repeat setup steps until it passes.

**Final Completion Signal:**
12. When verification passes and no further setup steps are needed, output exactly:
```
SETUP_COMPLETE  
```
and nothing else. This signals to the caller that setup is done.
"""

PROMPT_TEMPLATE = """
[COMMAND HISTORY]
{history}
"""

class TestAgent():
    def __init__(self):
        self.LLM = CoreAgent(model_id="claude-sonnet-4-20250514")
        self.tools = [{"type": "bash_20250124", "name": "bash"}]

    def step(self, environment: Environment):
        prompt = PROMPT_TEMPLATE.format(history=environment.history)
        response = self.LLM.query_tools(input_str=prompt, 
                                   tools=self.tools,
                                   system_prompt=SYSTEM_PROMPT
                                   )
        
        text = None
        for content in response.content:
            if content.type == "tool_use" and content.name == "bash":
                command = content.input.get("command")
            if content.type == 'text':
                text = content.text

        print(f"{response}\n")
        # if command:
        #     print(f"Agent current command: {command}")
        # else:
        #     print(f"Agent current command: {text}")
        action = Action(command=command, description=text)
        
        environment.execute(action)


    def run(self, environment: Environment, cycles=3):
        if cycles:
            for _ in range(cycles):
                self.step(environment)
        # else:
        #     while True:
        #         self.step(environment)


if __name__ == "__main__":
    env = Environment(repo_path="data/CSRBench100/storm", image_name="benchmark-image")
    agent = TestAgent()
    agent.run(env)
    env.log_environment_history(log_file='logs/test.jsonl')
