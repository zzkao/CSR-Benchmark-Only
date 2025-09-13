from core_agent import CoreAgent

SYSTEM_PROMPT = """
You are a strict evaluator of bash script executions.
You will be given:

* The full text of a bash script
* The full output produced by running it in a terminal

Your job is to decide **only** whether the script executed successfully or not.

* If the script executed fully without errors or failures, output exactly: `SUCCESS`
* If the script failed, had errors, or did not complete successfully, output exactly: `FAILED`

**Important rules**

* Do not explain your reasoning.
* Do not output anything except `SUCCESS` or `FAILED`.
* Assume any nonzero exit code, error messages, stack traces, command not found, or similar indications mean failure.
* If uncertain, default to `FAILED`.
"""

PROMPT_TEMPLATE = """
[BASH SCRIPT]
{bash_script}

[OUTPUT]
{output}
"""

class ScriptEvaluator():
    """
    Given a bash script and terminal output, determine if the script executed properly
    """
    def __init__(self):
        self.LLM = CoreAgent(model_id="claude-sonnet-4-20250514")
        self.name = "test_script_agent"

    def query(self, bash_script: str, output: str):
        prompt = PROMPT_TEMPLATE.format(bash_script=bash_script,
                                        output=output
                                        )
        message = self.LLM.query(input_str=prompt,
                                  system_prompt=SYSTEM_PROMPT
                                  )      
        
        eval = message.content[0].text
        if 'SUCCESS' in eval:
            return True
        else:
            return False
  

if __name__ == "__main__":
    agent = ScriptEvaluator()
    agent.query("ls", "data/")
