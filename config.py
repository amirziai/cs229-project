from typing import Dict

# global
experiment_name = 'exp1'
runs = 5
d_word = 300
lr = 0.0003
max_seq_len = 20
max_word_v_size = 100000


class Exp:
    def __init__(self, n_runs: int,
                 exp_name: str,
                 run_name: str,
                 experiment_parameters: Dict):
        self.runs = n_runs
        self.exp_name = exp_name
        self.run_name = run_name
        self.experiment_parameters = experiment_parameters


# per
experiments = [
    # Exp(n_runs=runs, exp_name=experiment_name, run_name='run1', experiment_parameters={
    #     'tasks': 'sst,mrpc',
    #     'weighting': 'uniform',  # epsilon, ucb
    #     # other stuff
    # }),
    Exp(n_runs=runs, exp_name=experiment_name, run_name='run2', experiment_parameters={
        'tasks': 'sst,sts-b',
        'weighting': 'uniform',  # epsilon, ucb
        # other stuff
    }),
]
