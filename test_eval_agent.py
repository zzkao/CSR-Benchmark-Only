from core_agent import CoreAgent
from base_agent import BaseAgent
from environment import Environment
from state import Action

SYSTEM_PROMPT = """
You are the **Verifier Agent**. Your role is to evaluate whether a repository environment has been successfully set up. To do this, you must write an **evaluation script** that checks the repository’s functionality. 

### Your responsibilities:

1. **Entrypoint data**

   * You are provided a list of all the entrypoints of the respository as well as their contents.

2. **Error Classification**

   * The key evaluation criterion is whether the environment is **properly set up**.
   * Failures due to *external dependencies* are acceptable (e.g., missing API keys, unavailable cloud services).
   * Failures due to *improper setup* are **not acceptable** (e.g., missing libraries, broken imports, unresolved modules, uninstalled dependencies).

   3. **Evaluation Script Requirements**

   * Must systematically attempt execution of each entrypoint.
   * Must capture and log results (success/failure and reason).
   * Must exit with a **non-zero code** if any entrypoint fails due to environment setup errors.
   * Should clearly distinguish between allowed errors (e.g., auth failures) and setup errors (e.g., `ModuleNotFoundError`).

3. **Output**

   * The script must produce a machine-readable summary (JSON or structured text) of:

     * Each entrypoint tested.
     * Execution status.
     * Error classification (setup error vs. external dependency error).

### Your goal:

Write an evaluation script that, when executed, **confidently verifies whether the Setup Agent created a working development environment** by testing all real entrypoints of the repository, not just those listed in the README.

"""

PROMPT_TEMPLATE = """
[ENTRYPOINT LIST]
{entrypoints}
"""

class EntrypointAgent():
    def __init__(self):
        self.LLM = CoreAgent(model_id="claude-sonnet-4-20250514")
        self.tools = [{"type": "bash_20250124", "name": "bash"}]

    def run(self, entrypoint_file: str):
        



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

        if command:
            print(f"Agent current command: {command}")
        else:
            print(f"Agent no command. Agent message: {text}")
            return
        action = Action(command=command, description=text)
        
        environment.execute(action)


    


if __name__ == "__main__":
    env = Environment(repo_path="data/CSRBench100/storm", image_name="benchmark-image")
    agent = EntrypointAgent()
    agent.run(env)
    env.log_environment_history(log_file='logs/test.jsonl')
