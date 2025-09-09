import argparse
import os
from environment import Environment
from test_agent_framework import TestAgent
from test_entrypoint_agent import EntrypointAgent
from test_eval_agent import TestScriptAgent
from datetime import datetime
import subprocess

parser = argparse.ArgumentParser(description='GSRBench100')
parser.add_argument('--repo', type=str, help='Repository link', required=True)
parser.add_argument('--docker', type=str, help='Docker image name', required=True)
parser.add_argument('--cycles', type=int, help='Number of cycles')
parser.add_argument('--keepdocker', action='store_true', help='Keep docker container after running benchmark')
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--script', action='store_true', help='Run agent to draft test script instead')
parser.add_argument('--keeprepo', action='store_true', help='Keep repo after running benchmark')
args = parser.parse_args()

REPO_LINK = args.repo
DOCKER_IMAGE_NAME = args.docker
NUM_CYCLES = args.cycles
KEEP_DOCKER = args.keepdocker
VERBOSE = args.verbose
SCRIPT = args.script
KEEP_REPO = args.keeprepo

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
results_file = f"./logs/results_{timestamp}.txt"

if REPO_LINK == "ALL":
    with open("./data/meta/CSRBench100.txt") as f:
        REPO_LINKS = [line.strip() for line in f]
else:
    REPO_LINKS = [f'{REPO_LINK}']


def run_agent(agent):
    if NUM_CYCLES:
        output, count = agent.run(env, cycles=NUM_CYCLES)
    else:
        output, count = agent.run(env)
    with open(results_file, "a") as f:
        f.write(f"{REPO_NAME}: {output}, Cycles: {count}\n")

for repo_link in REPO_LINKS:
    REPO_NAME= repo_link.rsplit('/', 1)[-1]
    env = Environment(repo_link, keep_docker=KEEP_DOCKER, image_name=DOCKER_IMAGE_NAME, verbose=VERBOSE)

    if SCRIPT:
        agent = EntrypointAgent()
        run_agent(agent)

    else:
        agent = TestAgent()
        run_agent(agent)
        env.run_test_scripts()

    if not KEEP_REPO:
        subprocess.run(["./clear_repos.sh"])