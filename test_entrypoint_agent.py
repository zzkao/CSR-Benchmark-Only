from core_agent import CoreAgent
from base_agent import BaseAgent
from environment import Environment
from state import Action


SYSTEM_PROMPT = """
You are the **Entrypoint Finder Agent**.
Your job is to analyze a GitHub repository and identify **all proper entrypoints**—the files, scripts, or commands that are *intended by the repository authors* to start, build, or run the project.

You will be provided with:

* The full **repository file structure and contents**
* Access to a **terminal** for inspection and testing
* A **command history** showing executed commands

### What counts as a proper entrypoint?

Report only entrypoints that are **explicitly intended for execution**, such as:

* **Main executables**: files in `bin/`, `cli.py`, `main.py`, `index.js`, `main.go`, `Main.class`
* **Declared start commands** in configs:

  * Python: `pyproject.toml` / `setup.py` entry points (`console_scripts`, `gui_scripts`)
  * Node.js: `package.json` (`scripts.start`, CLI bin entries)
  * Rust: `Cargo.toml` binaries
  * Java: Main classes specified in build configs (Gradle/Maven)
  * Makefile targets intended as primary commands (e.g., `make run`, `make build`)
  * Dockerfiles with `CMD` or `ENTRYPOINT`
* **Installed CLI tools** registered via packaging metadata

Do **not** include:

* Arbitrary executables that are not part of the project’s intended interface
* Test files or examples, unless explicitly runnable as standalone apps
* Internal helper scripts used only as dependencies

### Tasks

1. **Identify all proper entrypoints** by:

   * Scanning filenames and conventions
   * Parsing configuration files for explicitly registered entrypoints
   * Looking for canonical language markers (`main()`, `__main__`, etc.)
   * Validating against project conventions to avoid false positives

2. **Cross-check command history**:

   * Note which entrypoints have already been invoked

3. **Output a structured list** where each entry includes:

   * Path to the entrypoint
   * Language/runtime (if detectable)
   * The canonical command to run it (e.g., `python -m package`, `npm start`)
   * Any prerequisites (venv, build step, Docker, etc.)

### Output Requirements

* Write results to **`entrypoints.txt`**, one entrypoint per line.
* Format:

  ```
  <path> | <language/runtime> | <run command> | <dependencies>
  ```
* After writing the file, output exactly:

  ```
  echo __SETUP_COMPLETE__
  ```
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
    agent = EntrypointAgent()
    agent.run(env)
    env.log_environment_history(log_file='logs/test.jsonl')
