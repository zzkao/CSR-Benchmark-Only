from core_agent import CoreAgent
from base_agent import BaseAgent
from environment import Environment
from state import Action

SYSTEM_PROMPT = """
You are an assistant that helps verify software setup. Your role is to evaluate whether a repository environment has been successfully set up. To do this, you must write an **evaluation script** that checks the repository’s functionality.

### Your responsibilities:

1. **Repository Exploration**

   * You are given access to the full repository and a terminal.
   * You may freely inspect files, directories, and project metadata (e.g., `setup.py`, `requirements.txt`, `package.json`, `pyproject.toml`, Makefiles, Dockerfiles).
   * Do not rely solely on the README; instead, you must dynamically identify all possible **entrypoints**.

2. **Entrypoint Detection**

   * Entrypoints may include (but are not limited to):

     * CLI scripts defined in `setup.py`, `pyproject.toml`, or `package.json`.
     * Main scripts in `src/`, `bin/`, or `scripts/`.
     * Files containing `if __name__ == "__main__":`.
     * Declared commands in Makefiles or shell scripts.
   * Your evaluation script must attempt to run **every entrypoint** you identify.

3. **Error Classification**

   * The key evaluation criterion is whether the environment is **properly set up**.
   * Failures due to *external dependencies* are acceptable (e.g., missing API keys, unavailable cloud services).
   * Failures due to *improper setup* are **not acceptable** (e.g., missing libraries, broken imports, unresolved modules, uninstalled dependencies).

4. **Evaluation Script Requirements**

   * Must systematically attempt execution of each entrypoint.
   * Must capture and log results (success/failure and reason).
   * Must exit with a **non-zero code** if any entrypoint fails due to environment setup errors.
   * Should clearly distinguish between allowed errors (e.g., auth failures) and setup errors (e.g., `ModuleNotFoundError`).
   * You must be able to run the script by running the command `python TEST_SCRIPT.py`

5. **Output**

   * The script must produce a machine-readable summary (JSON or structured text) of:

     * Each entrypoint tested.
     * Execution status.
     * Error classification (setup error vs. external dependency error).

6. **Final Completion Signal:**

  * When you are done writing the script, output exactly: `echo __SETUP_COMPLETE__` and nothing else. This signals to the caller that setup is done.

### Your goal:

Write an evaluation script that, when executed, **confidently verifies whether the Setup Agent created a working development environment** by testing all real entrypoints of the repository, not just those listed in the README.

"""

PROMPT_TEMPLATE = """
[COMMAND HISTORY]
{history}
"""

class TestScriptAgent():
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

        if command:
            print(f"Agent current command: {command}")
        else:
            print(f"Agent finished commands. Final message: {text}")
            return
        action = Action(command=command, description=text)
        
        environment.execute(action)


    def run(self, environment: Environment, cycles=None):
        count = 0
        if cycles:
            for _ in range(cycles):
                count += 1
                try:
                    self.step(environment)
                    if "__SETUP_COMPLETE__" in environment.history[-1].output:
                        return 1, count
                except:
                    return 0, count
        else:
            while True:
                count += 1
                try: 
                    self.step(environment)
                    if "__SETUP_COMPLETE__" in environment.history[-1].output:
                        return 1, count
                except:
                    return 0, count
        return 0, count
        


if __name__ == "__main__":
    env = Environment(repo_path="data/CSRBench100/storm", image_name="benchmark-image")
    agent = TestScriptAgent()
    agent.run(env)
    env.log_environment_history(log_file='logs/test.jsonl')
