from typing import Optional

class Action:
    def __init__(self, command: str, agent_name: str, description: Optional[str] = None, ):
        self.command = command
        self.description = description
        self.agent_name = agent_name
    
    def to_dict(self):
        return {"command": self.command, "description": self.description}

    def __str__(self):
        parts = []

        if self.command.strip():
            parts.append(f"command:\n{self._indent(self.command)}")
        if self.description and self.description.strip():
            parts.append(f"description:\n{self._indent(self.description)}")

        return "\n".join(parts)

    @staticmethod
    def _indent(text, spaces=2):
        return "\n".join(" " * spaces + line for line in text.splitlines())

# class BashOutput:
#     def __init__(self, stdout: str, stderr: str, exit_code: int):
#         self.stdout = stdout
#         self.stderr = stderr
#         self.exit_code = exit_code

#     def to_dict(self):
#         return {"stdout": self.stdout, "stderr": self.stderr, "exit_code": self.exit_code}

#     def __str__(self):
#         parts = []

#         if self.stdout.strip():
#             parts.append(f"stdout:\n{self._indent(self.stdout)}")
#         if self.stderr.strip():
#             parts.append(f"stderr:\n{self._indent(self.stderr)}")

#         parts.append(f"exit code: {self.exit_code}")
#         return "\n".join(parts)

#     @staticmethod
#     def _indent(text, spaces=2):
#         return "\n".join(" " * spaces + line for line in text.splitlines())

class State:
    def __init__(self, action: Action, output: str):
        self.action = action
        self.output = output
        self.eval = None
    
    def to_dict(self):
        if self.eval:
            return {"action": self.action.to_dict(), "output": self.output, "eval": self.eval}
        else:
            return {"action": self.action.to_dict(), "output": self.output}
    
    def set_eval(self, eval):
        self.eval = eval

    def __str__(self):
        if self.eval:
            return (
                "Action:\n"
                f"{self._indent(str(self.action))}\n"
                "Output:\n"
                f"{self._indent(self.output)}\n"
                "Evaluation:\n"
                f"{self._indent(self.eval)}"
            )
        else:
            return (
                "Action:\n"
                f"{self._indent(str(self.action))}\n"
                "Output:\n"
                f"{self._indent(self.output)}"
            )
    
    def __repr__(self):
        return self.__str__()

    @staticmethod
    def _indent(text, spaces=2):
        return "\n".join(" " * spaces + line for line in text.splitlines())
    