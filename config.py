from typing import Dict, List

# global
experiment_name = 'exp1'
n_runs = 5
d_word = 300
lr = 0.0003
max_seq_len = 20
max_word_v_size = 100000
n_jobs = 8

ExperimentParam = Dict[str, str]


class Experiment:
    def __init__(self, run_name: str, params: ExperimentParam):
        self.run_name = run_name
        self.params = params


# experiment type
Experiments = List[Experiment]

# per
experiments: Experiments = [
    # Exp(n_runs=runs, exp_name=experiment_name, run_name='run1', experiment_parameters={
    #     'pretrain_tasks': '"sst,mrpc"',
    #     'weighting': 'uniform',  # epsilon, ucb
    #     # other stuff
    # }),
    # {
    #     'pretrain_tasks': 'sst,sts-b',
    #     'weighting': 'uniform',  # epsilon, ucb
    #     # other stuff
    # },
    # Experiment(run_name='run4', params={
    #     'pretrain_tasks': '"sst,sts-b"',
    #     'weighting': 'uniform',  # epsilon, ucb
    #     # other stuff
    # })
    Experiment(run_name='run6', params={
        'pretrain_tasks': '"sst,qqp"',
        'weighting': 'proportional',  # epsilon, ucb
        # other stuff
    })
]
