import os
import json
from base_agent import BaseAgent
from state import *
from environment import *

default_stages = [
            '# Environment Setup / Requirement / Installation',
            '# Data / Checkpoint / Weight Download (URL)',
            '# Training',
            '# Inference / Demonstration',
            '# Testing / Evaluation'
        ]

class Benchmark:
    def __init__(self, environment: Environment, stages=None, results_dir='results'):
        self.environment = environment
        self.stages = stages if stages is not None else default_stages
        self.results_dir = results_dir
        os.makedirs(self.results_dir, exist_ok=True)

    def run(self, agent: BaseAgent):
        repo_name = os.path.basename(self.working_dir.rstrip("/"))

        for stage in self.stages:
            result = agent.step()
            result_log = {}

        
        result_log['staged'] = result

        self._log_results(repo_name, result_log, readme_path, working_dir, ['unstaged'])

    def _log_results(self, repo_name: str, results: dict, readme_path: str, working_dir: str, stages: list):
        log_object = {
            "repo_name": repo_name,
            "working_dir": working_dir,
            "readme_path": readme_path,
            "stages": stages,
            "results": results
        }

        output_path = os.path.join(self.results_dir, f"{repo_name}.json")
        with open(output_path, 'w') as f:
            json.dump(log_object, f, indent=2)
        print(f"Saved results to {output_path}")

