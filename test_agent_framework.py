from core_agent import CoreAgent
from base_agent import BaseAgent
from environment import Environment
from state import Action

SYSTEM_PROMPT = """System message:
You are an assistant that helps execute software setup and usage instructions. The repository has already been cloned and you will be given:

1. The README of a repository (providing setup and usage details).
2. A chronological history of commands that have already been executed.

Your job is to:

1. Understand what the repository is about.
2. Infer what setup or usage steps have already been completed based on the command history.
3. Determine the next most logical step(s) to continue the setup or usage process.
4. Provide commands to set up the repository according to the README.
5. Whenever you provide a command, only provide one command at a time.
6. Avoid repeating commands that have already been executed successfully.

If information is missing from the README, make reasonable assumptions but state them explicitly.
"""

PROMPT_TEMPLATE = """
[README]
{readme}

[COMMAND HISTORY]
{history}
"""

class TestAgent():
    def __init__(self):
        self.LLM = CoreAgent(model_id="claude-sonnet-4-20250514")
        self.tools = [{"type": "bash_20250124", "name": "bash"}]

    def step(self, environment: Environment):
        prompt = PROMPT_TEMPLATE.format(readme=self.readme_content, history=environment.history)
        response = self.LLM.query_tools(input_str=prompt, 
                                   tools=self.tools,
                                   system_prompt=SYSTEM_PROMPT
                                   )
        for content in response.content:
            if content.type == "tool_use" and content.name == "bash":
                command = content.input.get("command")
            if content.type == 'text':
                text = content.text

        action = Action(command=command, description=text)
        print(f"ACTION: {action}")
        
        state = environment.execute(action)
        print(f"STATE: {state}")



    def run(self, environment: Environment):
        with open(f"{environment.repo_path}/README.md", "r") as f:  # "r" = read mode
            self.readme_content = f.read()
        
        for _ in range(10):
            self.step(environment)
        
        print(environment.history)



if __name__ == "__main__":
    env = Environment(repo_path="data/CSRBench100/storm", image_name="benchmark-image")
    agent = TestAgent()
    agent.run(env)