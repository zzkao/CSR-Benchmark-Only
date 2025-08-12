# CSR-Benchmark-Only

Refactored and reorganized CSR-Bench without the default frameworks.

## state.py
Contains data classes:
- `Action`, which represents the command the LLM wishes to run
- `BashOutput`, which represents the shell output after running the command
- `State`, which contains `Action` and `BashOutput`

## command_executor.py
This file handles the execution of commands by starting and running commands in Docker. The way that outputs are recieved is by writing to files in a `tmp/` folder that is shared between your local and Docker. These temp files have the unique docker container name attached so that there are no race conditions between containers and removed when the environment terminates.

## environment.py
Contains the `Environment` class, which keeps a list of `State` as history and `CommandExecutor` to change the environment. Additionally, it creates a Docker container that isolates the LLM environment.

## core_agent.py
Incomplete. Meant to be an abstract class for future agent frameworks to be build off

## bench.py
Incomplete. Potentially will be a layer on top of environment that can handle the runs of all benchmarking tests given an Agent

## TODO
1. Design and implement bench.py (optional)
2. Compatability with Openhand agents
3. Agent implementations and core_agent.py
