import os
from typing import List

from jiant.__main__ import main as jiant_main
import config
# from utils import log


class Experiment:
    def __init__(self, experiment_list: List):
        self.experiment_list = experiment_list

        os.environ['JIANT_PROJECT_PREFIX'] = 'jiant/'
        os.environ['JIANT_DATA_DIR'] = f'jiant/data/'
        os.environ['WORD_EMBS_FILE'] = f'jiant/embeddings/crawl-300d-2M.vec'

    @staticmethod
    def _create_conf_file(vals: dict, file_name: str):
        lines = [l for l in open('jiant/config/tutorial.conf')]
        out = []
        for line in lines:
            for key, val in vals.items():
                if line.startswith(key):
                    out.append(f'{key} = {val}\n')
                else:
                    out.append(line)

        with open(file_name, 'w') as f:
            f.writelines(out)

    @staticmethod
    def _run(conf_file: str, exp_name: str, run_name: str):
        conf = [
            '--config_file', conf_file,
            '--overrides', f"exp_name = {exp_name}, run_name = {run_name}"
        ]
        jiant_main(conf)

    @staticmethod
    def _parse_results(exp_name: str, run_name: str):
        pass  # TODO

    def run(self):
        for experiment in self.experiment_list:
            for run_idx in range(experiment.runs):
                self._create_conf_file(experiment.vals, experiment.conf_file)
                self._run(experiment.conf_file, experiment.exp_name, experiment.run_name)
                results = self._parse_results(experiment.exp_name, experiment.run_name)

            # TODO: combine the runs

        # TODO: combine all the experiments into a single csv


if __name__ == '__main__':
    path = '/Users/aziai/jiant'
    exp = Experiment(config.experiments)
    exp.run()
