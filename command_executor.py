import pexpect
from state import *

class CommandExecutor:
    def __init__(self, timeout=900):
        # Initialize the pexpect child process
        self.child = pexpect.spawn('/bin/bash', encoding='utf-8', timeout=timeout, echo=False)
        self.set_prompt()

    def set_prompt(self):
        # Set a unique prompt to ensure clear command boundaries
        unique_prompt = 'CMD_EXECUTOR_PROMPT>'
        self.child.sendline(f'export PS1="{unique_prompt}"')
        self.child.expect(unique_prompt)

    def execute(self, action: Action, directory=None):
        command = action.command
        command = ' && '.join(line.strip().replace('\\', ' ') for line in command.splitlines())

        stdout_file = '/tmp/cmd_stdout.txt'
        stderr_file = '/tmp/cmd_stderr.txt'
        exit_code_file = '/tmp/cmd_exit.txt'

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
            if directory:
                # Change to the specified directory
                self.child.sendline(f'cd "{directory}"')
                self.child.expect('CMD_EXECUTOR_PROMPT>')

            # Execute the command with stdout and stderr directed to respective files
            self.child.sendline(f'{command} >{stdout_file} 2>{stderr_file}; echo $? >{exit_code_file}')
            # Wait for the command to complete and capture the output including the exit code
            self.child.expect('CMD_EXECUTOR_PROMPT>')
                
            # Read stdout and stderr from their respective files
            # Handle exceptions if the files are not found (set to empty strings)
            stdout_output, stderr_output, exit_code = read_std()

            return BashOutput(stdout=stdout_output, stderr=stderr_output, exit_code=exit_code)
        
        except pexpect.exceptions.TIMEOUT:
            stdout_output, stderr_output = read_std()
            return BashOutput(stdout=stdout_output, stderr=stderr_output + "\nCommand timed out", exit_code=124)
        
        
        except pexpect.exceptions.EOF:
            stdout_output, stderr_output = read_std()
            return BashOutput(stdout=stdout_output, stderr=stderr_output+ "\nEOF", exit_code=1)
        
        except Exception as e:
            stdout_output, stderr_output = read_std()
            return BashOutput(stdout=stdout_output, stderr=stderr_output + "\n" + str(e), exit_code=1)
        
    def close(self):
        # Ensure the child process is properly terminated
        self.child.sendline('exit')
        self.child.close()


if __name__ == "__main__":
    # Example usage
    executor = CommandExecutor(timeout=30)

    # Create a loop and prompt the user for commands
    while True:
        command = input('Enter a command to execute (or "exit" to quit): ')
        if command.lower() == 'exit':
            break
        action = Action(command=command)
        execution_ret = executor.execute(action)
        print(State(action, execution_ret))

    # Close the session when done
    executor.close()

