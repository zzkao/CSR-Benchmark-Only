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

        # No -t for automation
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
        


    def execute(self, action: Action) -> BashOutput:
        """Executes an action and captures stdout, stderr, and exit code."""
        command = action.command
        command = ' && '.join(line.strip().replace('\\', ' ')
                              for line in command.splitlines())

        stdout_file = './tmp/cmd_stdout.txt'
        stderr_file = './tmp/cmd_stderr.txt'
        exit_code_file = './tmp/cmd_exit.txt'

        open(exit_code_file, 'w').close()
        open(stdout_file, 'w').close()
        open(stderr_file, 'w').close()

        def read_std():
            with open(stdout_file, 'r') as file:
                stdout_output = file.read().strip()
            with open(stderr_file, 'r') as file:
                stderr_output = file.read().strip()
            with open(exit_code_file, 'r') as file:
                exit_code = file.read().strip()
            
            return stdout_output.replace("\x00", ""), stderr_output.replace("\x00", ""), exit_code.replace("\x00", "")

        try:
            print(self.child.before)
            self.child.sendline(
                f'{command} >{stdout_file} 2>{stderr_file}; echo $? >{exit_code_file}'
            )

            self.child.expect(self.prompt)
            stdout_output, stderr_output, exit_code = read_std()

            return BashOutput(stdout=stdout_output, stderr=stderr_output, exit_code=exit_code)

        except pexpect.exceptions.TIMEOUT:
            stdout_output, stderr_output, _ = read_std()
            return BashOutput(stdout=stdout_output, stderr=stderr_output + "\nCommand timed out", exit_code="124")

        except pexpect.exceptions.EOF:
            stdout_output, stderr_output, _ = read_std()
            return BashOutput(stdout=stdout_output, stderr=stderr_output + "\nEOF", exit_code="1")

        except Exception as e:
            stdout_output, stderr_output, _ = read_std()
            return BashOutput(stdout=stdout_output, stderr=stderr_output + f"\n{str(e)}", exit_code="1")

    def close(self):
        """Close the pexpect child process."""
        if self.child.isalive():
            self.child.sendline('exit')
            self.child.close()
    
