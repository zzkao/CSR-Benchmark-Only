import argparse
import os
from environment import Environment
from test_agent_framework import TestAgent

parser = argparse.ArgumentParser(description='GSRBench100')
parser.add_argument('--repo', type=str, help='Repository link', required=True)
parser.add_argument('--docker', type=str, help='Docker image name', required=True)
parser.add_argument('--cycles', type=int, help='Number of cycles')
args = parser.parse_args()

REPO_LINK = args.repo
DOCKER_IMAGE_NAME = args.docker
NUM_CYCLES = args.cycles

if REPO_LINK == "ALL":
    with open("./data/meta/CSRBench100.txt") as f:
        REPO_LINKS = [line.strip() for line in f]
else:
    REPO_LINKS = [f'{REPO_LINK}']

for repo_link in REPO_LINKS:
    REPO_NAME= repo_link.rsplit('/', 1)[-1]
    env = Environment(repo_link, DOCKER_IMAGE_NAME)
    agent = TestAgent()

    if NUM_CYCLES:
        agent.run(env, cycles=NUM_CYCLES)
    else:
        agent.run(env)
    env.log_environment_history()