import os
from typing import List

import pandas as pd
from jiant.__main__ import main as jiant_main
import config
# from utils import log


class Experiment:
    def __init__(self,
                 experiment_list: List[config.Exp],
                 conf_path: str = 'configs/',
                 results_path: str = 'results/',
                 jiant_path: str = 'jiant/',
                 base_config_path: str = 'jiant/config/tutorial.conf'
                 ):
        self.experiment_list = experiment_list
        self.conf_path = conf_path
        self.results_path = results_path
        self.jiant_path = jiant_path
        self.base_config_path = base_config_path

        os.makedirs(self.conf_path, exist_ok=True)
        os.makedirs(self.results_path, exist_ok=True)
        os.environ['JIANT_PROJECT_PREFIX'] = self.jiant_path
        os.environ['JIANT_DATA_DIR'] = f'{self.jiant_path}data/'
        os.environ['WORD_EMBS_FILE'] = f'{self.jiant_path}embeddings/crawl-300d-2M.vec'

    def _create_conf_file(self, vals: dict, file_name: str):
        lines = [l for l in open(f'{self.jiant_path}{self.base_config_path}')]
        out = []
        for line in lines:

            k = None
            for key in vals:
                if line.startswith(key):
                    k = key
                    break

            if k is not None:
                out.append(f'{k} = {vals[k]}\n')
            else:
                out.append(line)

        with open(file_name, 'w') as f:
            f.writelines(out)

    @staticmethod
    def _run(conf_file: str, exp_name: str, run_name: str, run_idx: int):
        conf = [
            '--config_file', conf_file,
            '--overrides', f"exp_name = {exp_name}, run_name = {run_name}_{run_idx}"
        ]
        jiant_main(conf)

    def _parse_results(self, exp_name: str, run_name: str, run_idx: int) -> pd.DataFrame:
        file_path_log = f'{self.jiant_path}{exp_name}/{run_name}_{run_idx}/log.log'
        out = []

        first = None
        epoch_idx = 0
        for l in open(file_path_log):
            if 'f1' in l and '_acc_' not in l and 'validation' in l:
                parts = l.replace('\n', '').split(':')[3:]
                task = parts[0].replace('_f1', '').strip()

                if first is None:
                    first = task
                if task == first:
                    epoch_idx += 1
                train = float(parts[2].split()[0])
                val = float(parts[-1])
                out.append((epoch_idx, task, train, val))

        return pd.DataFrame(out, columns=['epoch_idx', 'task', 'train', 'val'])

    @staticmethod
    def _write_csv_out(df: pd.DataFrame, file_path: str) -> None:
        df.to_csv(file_path, index=False)

    @staticmethod
    def _append_experiment_metadata(df: pd.DataFrame,
                                    experiment: config.Exp,
                                    run_idx: int) -> pd.DataFrame:
        df['exp_name'] = experiment.exp_name
        df['run_name'] = experiment.run_name
        df['run_idx'] = run_idx

        for param, val in experiment.experiment_parameters.items():
            df[param] = val
        return df

    def run(self):
        file_path_results = f'{self.results_path}/{self.experiment_list[0].exp_name}.csv'
        results_overall = pd.DataFrame()

        for experiment in self.experiment_list:
            # TODO: clean up the paths
            conf_file = f'{self.jiant_path}jiant/config/{experiment.exp_name}_{experiment.run_name}.conf'

            for run_idx in range(experiment.runs):

                self._create_conf_file(experiment.experiment_parameters, conf_file)
                self._run(conf_file, experiment.exp_name, experiment.run_name, run_idx)
                results = self._parse_results(experiment.exp_name, experiment.run_name, run_idx)
                results = self._append_experiment_metadata(results, experiment, run_idx)
                results_overall = pd.concat([results_overall, results])
                # overwrite the results after each run
                self._write_csv_out(results_overall, file_path_results)


if __name__ == '__main__':
    exp = Experiment(config.experiments)
    exp.run()
