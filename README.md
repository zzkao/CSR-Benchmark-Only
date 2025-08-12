# CSR-Benchmark-Only

Refactored and reorganized CSR-Bench without the default frameworks.

## state.py
Contains data classes:
- `Action`, which represents the command the LLM wishes to run
- `BashOutput`, which represents the shell output after running the command
- `State`, which contains `Action` and `BashOutput`

## command_executor.py
This file handles the execution of commands by starting and running commands in a subshell. This file was updated from the original CSR-Bench and correctly reads exit code without needing an LLM to interpret `stdout` and `stderr`

## environment.py
Contains the `Environment` class, which keeps a list of `State` as history and `CommandExecutor` to change the environment

## core_agent.py
Incomplete. Meant to be an abstract class for future agent frameworks to be build off

## bench.py
Incomplete. Potentially will be a layer on top of environment that can handle the runs of all benchmarking tests given an Agent

## TODO
1. Design and implement bench.py (optional)
2. Docker containerization
3. Compatability with Openhand agents
4. Agent implementations and core_agent.py
