import pexpect
from state import *

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
        """Set a unique prompt to reliably detect command completion."""
        unique_prompt = 'CMD_EXECUTOR_PROMPT>'
        self.child.sendline(f'export PS1="{unique_prompt}"; echo READY')
        self.child.expect('READY')
        self.child.expect(unique_prompt)

        self.prompt = unique_prompt
        
    def execute(self, action: Action):
        """Executes an action and captures stdout, stderr, and exit code."""
        command = action.command
        command = ' && '.join(line.strip().replace('\\', ' ')
                              for line in command.splitlines())

        # absolute temp file paths inside docker
        stdout_file = f'/workspace/tmp/cmd_stdout{self.container_name}.txt'
        stderr_file = f'/workspace/tmp/cmd_stderr{self.container_name}.txt'
        exit_code_file = f'/workspace/tmp/cmd_exit{self.container_name}.txt'

        try:
            self.child.sendline(
                f'{command} >{stdout_file} 2>{stderr_file}; echo $? >{exit_code_file}'
            )

            self.child.expect(self.prompt)
            return 1
        
        except Exception as e:
            return e
        # except pexpect.exceptions.TIMEOUT:
        #     stdout_output, stderr_output, _ = read_std()
        #     return BashOutput(stdout=stdout_output, stderr=stderr_output + "\nCommand timed out", exit_code="124")

        # except pexpect.exceptions.EOF:
        #     stdout_output, stderr_output, _ = read_std()
        #     return BashOutput(stdout=stdout_output, stderr=stderr_output + "\nEOF", exit_code="1")

        # except Exception as e:
        #     stdout_output, stderr_output, _ = read_std()
        #     return BashOutput(stdout=stdout_output, stderr=stderr_output + f"\n{str(e)}", exit_code="1")

    def close(self):
        """Close the pexpect child process."""
        if self.child.isalive():
            self.child.sendline('exit')
            self.child.close()
    
