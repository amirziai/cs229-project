from typing import Dict

ExperimentParam = Dict[str, str]


class Experiment:
    def __init__(self, run_name: str, params: ExperimentParam):
        self.run_name = run_name
        self.params = params
