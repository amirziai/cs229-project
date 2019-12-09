import os
from typing import List

import pandas as pd
from joblib import Parallel, delayed

import config_exp1 as config
import config as conf
from jiant.__main__ import main as jiant_main


# from utils import log


class Experiment:
    def __init__(self,
                 experiment_name: str,
                 experiments: config.Experiments,
                 n_runs: int,
                 n_jobs: int,
                 conf_path: str = 'configs/',
                 results_path: str = 'results/',
                 jiant_path: str = 'jiant/',
                 base_config_path: str = 'jiant/config/tutorial.conf'
                 ):
        self.experiment_name = experiment_name
        self.experiments = experiments
        self.n_runs = n_runs
        self.n_jobs = n_jobs

        self.conf_path = conf_path
        self.results_path = results_path
        self.jiant_path = jiant_path
        self.base_config_path = base_config_path

        os.makedirs(self.conf_path, exist_ok=True)
        os.makedirs(self.results_path, exist_ok=True)
        os.environ['JIANT_PROJECT_PREFIX'] = self.jiant_path
        os.environ['JIANT_DATA_DIR'] = f'{self.jiant_path}data/'
        os.environ['WORD_EMBS_FILE'] = f'{self.jiant_path}embeddings/crawl-300d-2M.vec'

    def _create_conf_file(self, experiment: conf.ExperimentParam, file_name: str):
        lines = [l for l in open(f'{self.jiant_path}{self.base_config_path}')]
        out = []
        for line in lines:

            k = None
            for key in experiment:
                if line.startswith(key):
                    k = key
                    break

            if k is not None:
                out.append(f'{k} = {experiment[k]}\n')
            else:
                out.append(line)

        with open(file_name, 'w') as f:
            f.writelines(out)

    @staticmethod
    def _run_jiant(conf_file: str, exp_name: str, run_name: str, run_idx: int):
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

    def _append_experiment_metadata(self,
                                    df: pd.DataFrame,
                                    experiment: conf.ExperimentParam,
                                    run_name: str,
                                    run_idx: int) -> pd.DataFrame:
        df['exp_name'] = self.experiment_name
        df['run_name'] = run_name
        df['run_idx'] = run_idx

        for param, val in experiment.items():
            df[param] = val
        return df

    def _run_exp(self,
                 experiment: config.Experiment,
                 run_idx: int) -> pd.DataFrame:
        run_name = experiment.run_name
        conf_file = f'{self.jiant_path}jiant/config/{self.experiment_name}_{run_name}.conf'  # TODO: cleanup paths
        self._create_conf_file(experiment.params, conf_file)
        self._run_jiant(conf_file, self.experiment_name, run_name, run_idx)
        results = self._parse_results(self.experiment_name, run_name, run_idx)
        results = self._append_experiment_metadata(results, experiment.params, run_name, run_idx)
        file_path_results = f'{self.results_path}/{self.experiment_name}_{run_name}_{run_idx}.csv'
        self._write_csv_out(results, file_path_results)
        return results

    def run(self):
        file_path_results_overall = f'{self.results_path}/{self.experiment_name}.csv'
        results_list: List[pd.DataFrame] = Parallel(n_jobs=self.n_jobs)(
            delayed(self._run_exp)(experiment, run_idx)
            for run_name_idx, experiment in enumerate(self.experiments)
            for run_idx in range(self.n_runs))

        results_overall = pd.concat(results_list)
        self._write_csv_out(results_overall, file_path_results_overall)


if __name__ == '__main__':
    exp = Experiment(config.experiment_name, config.experiments, config.n_runs, config.n_jobs)
    exp.run()
