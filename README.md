# CSR-Benchmark-Only
A refactored, reorganized version of CSR-Bench — stripped of default frameworks and built for flexibility in testing LLM command execution.

## Overview
This repository contains the core components needed to:

Isolate LLM environments inside Docker.

Track command execution history.

Provide a foundation for building and benchmarking custom agent frameworks.

## File Structure

### `state.py`

Defines the primary data structures:
- Action – Represents a command the LLM intends to run.
- State – Bundles together an Action and a string containing the bash output.

### `command_executor.py`

Executes commands within a Docker container.

- Output is captured by writing to temporary files inside a shared tmp/ directory.

- Temp files are container-specific (tagged with the unique container name) to avoid race conditions.

- Files are removed automatically when the environment shuts down.

### `environment.py`

Implements the Environment class:

- Maintains a history of State objects.

- Uses CommandExecutor to modify the container state.

- Automatically spins up an isolated Docker container for the LLM.

### `core_agent.py`

An abstract base class intended for building new agent frameworks.
Currently incomplete.

### `bench.py`

A planned orchestration layer for running full benchmark suites against a given Agent.
Currently incomplete.

## Building the Docker Container
To build the Docker image used for environment isolation:

```bash
./docker_setup.sh
```
> **Note:** The default Docker image is a placeholder. You can modify it as needed.

## Testing Docker Containerization of Environment
After building the Docker image, try running an enviornment in a Docker container using that image: 
```bash
python environment.py
```

## Test Dummy Agent Framework
Once docker works, paste your `ANTHROPIC_API_KEY` into `.env` try running 
```bash
python main.py --repo https://github.com/stanford-oval/storm --docker benchmark-image --cycles 75
```
to run the agent for a maximum of 75 cycles. The agent will automatically stop when setup is deemed complete and the log of the command history will be in `logs/`. Furthermore, a `results` file will be created to track the repos that were benchmarked, whether it was successful, and the number of cycles that were ran.


## TODO
- Implement bench.py (optional, but recommended for automation).

- Add compatibility with Openhand agents.

- Develop agent implementations based on core_agent.py.

- Stopping condition for basic test agent and bug fixes






