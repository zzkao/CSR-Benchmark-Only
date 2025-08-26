import pexpect
from state import *
import re

class CommandExecutor:
    """
    Executes commands in an already running Docker container using pexpect.
    Captures stdout, stderr, and exit code.
    """

    def __init__(self, container_name: str, timeout=900):
        self.container_name = container_name
        self.timeout = timeout
        
        cmd = f"docker exec -it {container_name} /bin/bash"
        self.child = pexpect.spawn(cmd, encoding='utf-8', timeout=timeout, echo=False)
        self._set_prompt()

    def _set_prompt(self):
        self.prompt = "__CMD_EXECUTOR_PROMPT__$ "  # include a trailing space for clarity
        prompt_escaped = re.escape(self.prompt)

        # Set a simple, unique prompt (separate commands; don't chain)
        self.child.sendline(f'export PS1="{self.prompt}"')
        # Disable remote echo so our typed commands don't appear in output
        self.child.sendline('stty -echo')

        # Sync & wait for the new prompt to appear
        self.child.sendline('echo READY')
        self.child.expect_exact('READY')
        self.child.expect_exact(self.prompt)

        # Store for later expects
        self._prompt_exact = self.prompt
        self._prompt_regex = prompt_escaped

    def _flush_until_prompt(self):
        """Ensure buffer is clean before starting a new command."""
        self.child.sendline('echo __SYNC__')
        self.child.expect_exact('__SYNC__')
        self.child.expect_exact(self._prompt_exact)

    def execute(self, action):
        command = getattr(action, 'command', None)
        if not command:
            return {"output": "", "exit_code": 1}

        self._flush_until_prompt()

        marker = "__EXIT__MARKER__"
        
        # Send multi-line command safely
        safe_command = f"{{\n{command}\n}}; printf '\\n{marker}%d{marker}\\n' \"$?\""

        self.child.sendline(safe_command)
        self.child.expect_exact(self._prompt_exact)

        raw = self.child.before.strip("\r\n")

        # Remove the echoed command if needed
        if raw.startswith(command):
            raw = raw[len(command):].lstrip()

        # Find exit code marker
        start = raw.rfind(marker)
        end = raw.rfind(marker, start + len(marker)) if start != -1 else -1

        output = raw
        if start != -1 and end != -1 and end > start:
            output = raw[:start].rstrip("\r\n")
        
        clean_output = self._clean_output(output)
        # print(clean_output)
        return clean_output

    def _clean_output(self, text: str) -> str:
        # remove exit markers
        text = re.sub(r"__EXIT__MARKER__\d+__EXIT__MARKER__", "", text)
        # remove ANSI escape sequences
        text = re.sub(r"\x1B\[[0-?]*[ -/]*[@-~]", "", text)
        # strip leading/trailing whitespace and collapse blank lines
        lines = [line for line in text.splitlines() if line.strip()]

        # handle carriage return progress bars: keep only last line after \r
        cleaned_lines = []
        for line in lines:
            if "\r" in line:
                line = line.split("\r")[-1]  # keep only final overwrite
            if line.strip() and "> " not in line:
                cleaned_lines.append(line)

        return "\n".join(cleaned_lines)

    def close(self):
        """Close the pexpect child process."""
        if self.child.isalive():
            self.child.sendline('exit')
            self.child.close()
    
