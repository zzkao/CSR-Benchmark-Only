from core_agent import CoreAgent
from base_agent import BaseAgent
from environment import Environment
from state import Action


SYSTEM_PROMPT = """
**System Prompt (Proper Entrypoints Only, with Strong Completion Signal)**

You are the **Entrypoint Finder Agent**.
Your job is to analyze a GitHub repository and identify **all proper entrypoints**—the files, scripts, or commands that are *intended by the repository authors* to start, build, or run the project.

You will be provided with:

* The full **repository file structure and contents**
* Access to a **terminal** for inspection and testing
* A **command history** showing executed commands

### What counts as a proper entrypoint?

Include only entrypoints that are **explicitly intended** for execution, such as:

* **Main executables**: `cli.py`, `main.py`, `index.js`, `main.go`, `Main.class`, or files in `bin/`.
* **Declared start commands** in configs:

  * Python: `pyproject.toml`, `setup.py` (`console_scripts`, `gui_scripts`)
  * Node.js: `package.json` (`scripts.start`, bin entries)
  * Rust: `Cargo.toml` binaries
  * Java: Main classes defined in Gradle/Maven configs
  * Makefile: canonical targets (`make run`, `make build`)
  * Dockerfile: `CMD` or `ENTRYPOINT`
* **Installed CLI tools** registered via packaging metadata

Do **not** include:

* Arbitrary executables not part of the official interface
* Test files, examples, or helper scripts

### Tasks

1. **Identify all proper entrypoints** by scanning files, configs, and conventions.
2. **Cross-check with command history** and note which were already invoked.
3. **Output results** to a file named `entrypoints.txt`. Each line must follow the format:

   ```
   <path>
   ```

### Final Completion Signal (Mandatory)

* Once `entrypoints.txt` has been written successfully, you **must output exactly and only**:

  ```
  echo __SETUP_COMPLETE__
  ```
* Do not output explanations, logs, or extra text.
* This final echo is **required** for successful completion.
"""

PROMPT_TEMPLATE = """
[COMMAND HISTORY]
{history}
"""

class EntrypointAgent():
    def __init__(self):
        self.LLM = CoreAgent(model_id="claude-sonnet-4-20250514")
        self.tools = [{"type": "bash_20250124", "name": "bash"}]
        self.name = "entrypoint_agent"
        self.repeat = 0

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
        action = Action(command=command, description=text, agent_name=self.name)
        
        environment.execute(action)


    def run(self, environment: Environment, cycles=None):
        count = 0
        if cycles:
            for _ in range(cycles):
                count += 1
                try:
                    self.step(environment)
                    if "__SETUP_COMPLETE__" in environment.history[self.name][-1].output or self._check_loop(environment=environment):
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
    
    def _check_loop(self, environment: Environment):
        k = 5
        if len(environment.history[self.name]) >= k and all(environment.history[self.name][-1].action.command == environment.history[self.name][-i].action.command for i in range(2,k+1)):
            return True
        return False

        


if __name__ == "__main__":
    env = Environment(repo_path="data/CSRBench100/storm", image_name="benchmark-image")
    agent = EntrypointAgent()
    agent.run(env)
    env.log_environment_history(log_file='logs/test.jsonl')
