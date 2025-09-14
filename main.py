import argparse
from environment import Environment
from test_agent_framework_HARD import HardTestAgent
from test_agent_framework_EASY import EasyTestAgent
from test_entrypoint_agent import EntrypointAgent
from datetime import datetime
import subprocess
import traceback

parser = argparse.ArgumentParser(description='GSRBench100')
parser.add_argument('--repo', type=str, help='Repository link', required=True)
parser.add_argument('--docker', type=str, help='Docker image name', required=True)
parser.add_argument('--cycles', type=int, help='Number of cycles')
parser.add_argument('--keepdocker', action='store_true', help='Keep docker container after running benchmark')
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--agent', action='store_true', choices=['easy', 'hard', 'entrypoint'], help='Type of agent to run', required=True)
parser.add_argument('--keeprepo', action='store_true', help='Keep repo after running benchmark')
args = parser.parse_args()

REPO_LINK = args.repo
DOCKER_IMAGE_NAME = args.docker
NUM_CYCLES = args.cycles
KEEP_DOCKER = args.keepdocker
VERBOSE = args.verbose
AGENT = args.agent
KEEP_REPO = args.keeprepo

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
results_file = f"./logs/results_{timestamp}.txt"

with open("./data/meta/CSRBench100_full.txt") as f:
    repo_to_num = {line.strip(): i for i, line in enumerate(f, start=1)}


if REPO_LINK == "ALL":
    with open("./data/meta/CSRBench100_full.txt") as f:
        REPO_LINKS = [line.strip() for line in f]
else:
    REPO_LINKS = [f'{REPO_LINK}']

def run_agent(agent, env, REPO_NAME):
    try:
        if NUM_CYCLES:
            output, count = agent.run(env, cycles=NUM_CYCLES)
        else:
            output, count = agent.run(env)
        with open(results_file, "a") as f:
            f.write(f"{REPO_NAME}: {output}, Cycles: {count}")
    except Exception as e:
        with open(results_file, "a") as f:
            f.write(f"{REPO_NAME}: ERROR during agent run - {e}\n")
            f.write(traceback.format_exc() + "\n")

for repo_link in REPO_LINKS:
    repo_number = repo_to_num[repo_link]
    REPO_NAME = repo_link.rsplit('/', 1)[-1]

    try:
        env = Environment(
            repo_link,
            keep_docker=KEEP_DOCKER,
            image_name=DOCKER_IMAGE_NAME,
            verbose=VERBOSE
        )

        if AGENT == 'entrypoint':
            agent = EntrypointAgent()
            run_agent(agent, env, REPO_NAME)
        elif AGENT == 'HARD':
            agent = HardTestAgent()
            run_agent(agent, env, REPO_NAME)
            try:
                result = env.run_test_scripts(repo_number)
                with open(results_file, "a") as f:
                    f.write(f", Test Results: {result}\n")
            except Exception as e:
                with open(results_file, "a") as f:
                    f.write(f"{REPO_NAME}: ERROR during test scripts - {e}\n")
                    f.write(traceback.format_exc() + "\n")
        else:
            agent = EasyTestAgent(test_number=repo_number)
            run_agent(agent, env, REPO_NAME)
            try:
                result = env.run_test_scripts(repo_number)
                with open(results_file, "a") as f:
                    f.write(f", Test Results: {result}\n")
            except Exception as e:
                with open(results_file, "a") as f:
                    f.write(f"{REPO_NAME}: ERROR during test scripts - {e}\n")
                    f.write(traceback.format_exc() + "\n")

    except Exception as e:
        with open(results_file, "a") as f:
            f.write(f"{REPO_NAME}: FATAL ERROR - {e}\n")
            f.write(traceback.format_exc() + "\n")

    finally:
        if not KEEP_REPO:
            try:
                subprocess.run(["./clear_repos.sh"], check=True)
                env.close()
            except subprocess.CalledProcessError as e:
                with open(results_file, "a") as f:
                    f.write(f"{REPO_NAME}: ERROR closing repo - {e}\n")
