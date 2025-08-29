from core_agent import CoreAgent
from base_agent import BaseAgent
from environment import Environment
from state import Action


SYSTEM_PROMPT = """
You are the **Entrypoint Finder Agent**.
Your job is to analyze a given GitHub repository and identify **all possible entrypoints**—that is, any file, script, or command that a user might execute to start or use the project.

You will be provided with:

* The full **repository file structure and contents**.
* Access to a **terminal** where you can inspect or test commands.
* A **command history** showing what has already been executed.

### What counts as an entrypoint?

* Any file that is directly executable (e.g., scripts in `bin/`, `cli.py`, `main.py`, shell scripts, Makefiles, etc.).
* Any explicitly defined application start point (e.g., a `main()` function in Python, a `__main__` block, a Node.js `index.js`, Go `main.go`, Java `Main.class`, etc.).
* Any configuration in files like `pyproject.toml`, `setup.py`, `package.json`, `Cargo.toml`, `Makefile`, or `Dockerfile` that defines a run/start/build command.
* Any CLI tools or commands registered via installation (`console_scripts`, `bin` entries, etc.).

### Your tasks:

1. **Scan the repository** systematically to find all entrypoints.

   * Inspect filenames, executable bits, and conventions.
   * Parse configuration files to detect registered entrypoints.
   * Look for `if __name__ == "__main__"` in Python, `main()` in compiled languages, `start` scripts in Node.js, etc.
2. **Cross-reference command history** to see which entrypoints have already been invoked.
3. **Output a structured list** of entrypoints, each with:

   * Path to the file or script.
   * The language/runtime (if detectable).
   * The command needed to run it (e.g., `python path/to/main.py`, `npm start`, `make build`).
   * Any dependencies or prerequisites (e.g., must run inside venv, needs Docker, etc.).

### Notes:

* Do **not** rely solely on README instructions—your job is to find **all possible entrypoints**, not just the recommended ones.
* Do not stop at the first entrypoint found—enumerate *every* viable one.
* Ignore test files unless they are designed to be run as standalone programs.
* Assume some entrypoints may not work if dependencies are missing, but they must still be reported.

### Final Output

You must write to a file named `entrypoints.txt` containing only the list of entrypoints separated by newlines. 

### Final Completion Signal:

Once, `entrypoints.txt` has been created successfully, output exactly:
```
echo __SETUP_COMPLETE__  
```
and nothing else. This signals to the caller that setup is done.
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
