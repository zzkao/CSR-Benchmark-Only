from typing import Optional

class Action:
    def __init__(self, command: str, description: Optional[str] = None):
        self.command = command
        self.description = description

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

class BashOutput:
    def __init__(self, stdout: str, stderr: str, exit_code: int):
        self.stdout = stdout
        self.stderr = stderr
        self.exit_code = exit_code

    def __str__(self):
        parts = []

        if self.stdout.strip():
            parts.append(f"stdout:\n{self._indent(self.stdout)}")
        if self.stderr.strip():
            parts.append(f"stderr:\n{self._indent(self.stderr)}")

        parts.append(f"exit code: {self.exit_code}")
        return "\n".join(parts)

    @staticmethod
    def _indent(text, spaces=2):
        return "\n".join(" " * spaces + line for line in text.splitlines())

class State:
    def __init__(self, action: Action, output: BashOutput):
        self.action = action
        self.output = output

    def __str__(self):
        return (
            "Action:\n"
            f"{self._indent(str(self.action))}\n"
            "Output:\n"
            f"{self._indent(str(self.output))}"
        )

    @staticmethod
    def _indent(text, spaces=2):
        return "\n".join(" " * spaces + line for line in text.splitlines())
    