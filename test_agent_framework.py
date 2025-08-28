from core_agent import CoreAgent
from base_agent import BaseAgent
from environment import Environment
from state import Action

SYSTEM_PROMPT = """You are an assistant that helps execute software setup and usage instructions. You will be given:
1. A minimal installation of ubuntu with the repository already pulled
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
9. When installing packages, you must turn off the progress bar

**Detecting Setup Completion:**
10. Consider the setup complete when:
    * All required installation steps in the README have been executed successfully.
    * The repository's main functionality can be run without missing dependencies or configuration.
    * Any optional steps not essential for core functionality have been skipped or explicitly marked as such.

**Verification Step Before Declaring Completion:**
11. Before declaring completion, run an appropriate verification command (e.g., a test script, `--help` command, `make test`, or example run from README) to ensure the repository is functional.
12. If verification fails, continue troubleshooting and repeat setup steps until it passes.

**Final Completion Signal:**
13. When verification passes and no further setup steps are needed, output exactly:
```
echo __SETUP_COMPLETE__  
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
        self.name = "test_agent"

    def step(self, environment: Environment):
        prompt = PROMPT_TEMPLATE.format(history=environment.history[self.name])
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

        if command:
            print(f"Agent current command: {command}")
        else:
            print(f"Agent no command. Agent message: {text}")
        action = Action(command=command, description=text, name=self.name)
        
        environment.execute(action)


    def run(self, environment: Environment, cycles=None):
        count = 0
        if cycles:
            for _ in range(cycles):
                count += 1
                try:
                    self.step(environment)
                    if "__SETUP_COMPLETE__" in environment.history[self.name][-1].output:
                        return 1, count
                except:
                    return 0, count
        else:
            while True:
                count += 1
                try: 
                    self.step(environment)
                    if "__SETUP_COMPLETE__" in environment.history[self.name][-1].output:
                        return 1, count
                except:
                    return 0, count
        return 0, count
        


if __name__ == "__main__":
    env = Environment(repo_path="data/CSRBench100/storm", image_name="benchmark-image")
    agent = TestAgent()
    agent.run(env)
    env.log_environment_history(log_file='logs/test.jsonl')
