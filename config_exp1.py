from typing import List

from config import Experiment

# global
experiment_name = 'exp1'
n_runs = 5
d_word = 300
lr = 0.0003
max_seq_len = 20
max_word_v_size = 100000
n_jobs = 1

# experiment type
Experiments = List[Experiment]

# per
experiments: Experiments = [
    Experiment(run_name='run1', params={
        'pretrain_tasks': '"sst,mrpc"',
        'weighting_method': 'round_robin',
    }),
    Experiment(run_name='run2', params={
        'pretrain_tasks': '"sst,qqp"',
        'weighting_method': 'round_robin',
    }),
    Experiment(run_name='run3', params={
        'pretrain_tasks': '"sst,qnli"',
        'weighting_method': 'round_robin',
    }),
    Experiment(run_name='run4', params={
        'pretrain_tasks': '"sst,qqp"',
        'weighting_method': 'round_robin',
    }),
    Experiment(run_name='run5', params={
        'pretrain_tasks': '"sst,qqp,mrpc,qnli"',
        'weighting_method': 'round_robin',
    }),
    Experiment(run_name='run6', params={
        'pretrain_tasks': '"sst,qqp,qnli"',
        'weighting_method': 'round_robin',
    }),
    # now epsilon
    Experiment(run_name='run7', params={
        'pretrain_tasks': '"sst,mrpc"',
        'weighting_method': 'epsilon',
    }),
    Experiment(run_name='run8', params={
        'pretrain_tasks': '"sst,qqp"',
        'weighting_method': 'epsilon',
    }),
    Experiment(run_name='run9', params={
        'pretrain_tasks': '"sst,qnli"',
        'weighting_method': 'epsilon',
    }),
    Experiment(run_name='run10', params={
        'pretrain_tasks': '"sst,qqp"',
        'weighting_method': 'epsilon',
    }),
    Experiment(run_name='run11', params={
        'pretrain_tasks': '"sst,qqp,mrpc,qnli"',
        'weighting_method': 'epsilon',
    }),
    Experiment(run_name='run12', params={
        'pretrain_tasks': '"sst,qqp,qnli"',
        'weighting_method': 'epsilon',
    }),
    # now ucb
    Experiment(run_name='run13', params={
        'pretrain_tasks': '"sst,mrpc"',
        'weighting_method': 'ucb',
    }),
    Experiment(run_name='run14', params={
        'pretrain_tasks': '"sst,qqp"',
        'weighting_method': 'ucb',
    }),
    Experiment(run_name='run15', params={
        'pretrain_tasks': '"sst,qnli"',
        'weighting_method': 'ucb',
    }),
    Experiment(run_name='run16', params={
        'pretrain_tasks': '"sst,qqp"',
        'weighting_method': 'ucb',
    }),
    Experiment(run_name='run17', params={
        'pretrain_tasks': '"sst,qqp,mrpc,qnli"',
        'weighting_method': 'ucb',
    }),
    Experiment(run_name='run18', params={
        'pretrain_tasks': '"sst,qqp,qnli"',
        'weighting_method': 'ucb',
    }),
]
