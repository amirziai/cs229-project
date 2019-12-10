from typing import List

from config import Experiment

# global
experiment_name = 'exp5'
n_runs = 3
d_word = 300
lr = 0.0003
max_seq_len = 20
max_word_v_size = 100000
n_jobs = 16

# experiment type
Experiments = List[Experiment]

# per
experiments: Experiments = [
    Experiment(run_name='run1', params={
        'pretrain_tasks': '"sst,mrpc"',
        'transfer_paradigm': 'finetune',
        'weighting_method': 'round_robin',
        'dropout': '0.1',
        'input_module': 'bert-base-uncased',
        'classifier': 'log_reg',
        'optimizer': 'bert_adam',
        'sent_enc': 'none',
        'sep_embs_for_skip': '1',
    }),
    Experiment(run_name='run2', params={
        'pretrain_tasks': '"sst,qqp"',
        'transfer_paradigm': 'finetune',
        'weighting_method': 'round_robin',
        'dropout': '0.1',
        'input_module': 'bert-base-uncased',
        'classifier': 'log_reg',
        'optimizer': 'bert_adam',
        'sent_enc': 'none',
        'sep_embs_for_skip': '1',
    }),
    Experiment(run_name='run3', params={
        'pretrain_tasks': '"sst,qnli"',
        'weighting_method': 'round_robin',
        'dropout': '0.1',
        'input_module': 'bert-base-uncased',
        'transfer_paradigm': 'finetune',
        'classifier': 'log_reg',
        'optimizer': 'bert_adam',
        'sent_enc': 'none',
        'sep_embs_for_skip': '1',
    }),
    # epsilon
    Experiment(run_name='run4', params={
        'pretrain_tasks': '"sst,mrpc"',
        'weighting_method': 'epsilon',
        'dropout': '0.1',
        'input_module': 'bert-base-uncased',
        'transfer_paradigm': 'finetune',
        'classifier': 'log_reg',
        'optimizer': 'bert_adam',
        'sent_enc': 'none',
        'sep_embs_for_skip': '1',
    }),
    Experiment(run_name='run5', params={
        'pretrain_tasks': '"sst,qqp"',
        'weighting_method': 'epsilon',
        'dropout': '0.1',
        'input_module': 'bert-base-uncased',
        'transfer_paradigm': 'finetune',
        'classifier': 'log_reg',
        'optimizer': 'bert_adam',
        'sent_enc': 'none',
        'sep_embs_for_skip': '1',
    }),
    Experiment(run_name='run6', params={
        'pretrain_tasks': '"sst,qnli"',
        'weighting_method': 'epsilon',
        'dropout': '0.1',
        'input_module': 'bert-base-uncased',
        'transfer_paradigm': 'finetune',
        'classifier': 'log_reg',
        'optimizer': 'bert_adam',
        'sent_enc': 'none',
        'sep_embs_for_skip': '1',
    }),
    # ucb
    Experiment(run_name='run7', params={
        'pretrain_tasks': '"sst,mrpc"',
        'weighting_method': 'ucb',
        'dropout': '0.1',
        'input_module': 'bert-base-uncased',
        'transfer_paradigm': 'finetune',
        'classifier': 'log_reg',
        'optimizer': 'bert_adam',
        'sent_enc': 'none',
        'sep_embs_for_skip': '1',
    }),
    Experiment(run_name='run8', params={
        'pretrain_tasks': '"sst,qqp"',
        'weighting_method': 'ucb',
        'dropout': '0.1',
        'input_module': 'bert-base-uncased',
        'transfer_paradigm': 'finetune',
        'classifier': 'log_reg',
        'optimizer': 'bert_adam',
        'sent_enc': 'none',
        'sep_embs_for_skip': '1',
    }),
    Experiment(run_name='run9', params={
        'pretrain_tasks': '"sst,qnli"',
        'weighting_method': 'ucb',
        'dropout': '0.1',
        'input_module': 'bert-base-uncased',
        'transfer_paradigm': 'finetune',
        'classifier': 'log_reg',
        'optimizer': 'bert_adam',
        'sent_enc': 'none',
        'sep_embs_for_skip': '1',
    }),
    # proportional
    Experiment(run_name='run10', params={
        'pretrain_tasks': '"sst,mrpc"',
        'weighting_method': 'proportional',
        'dropout': '0.1',
        'input_module': 'bert-base-uncased',
        'transfer_paradigm': 'finetune',
        'classifier': 'log_reg',
        'optimizer': 'bert_adam',
        'sent_enc': 'none',
        'sep_embs_for_skip': '1',
    }),
    Experiment(run_name='run11', params={
        'pretrain_tasks': '"sst,qqp"',
        'weighting_method': 'proportional',
        'dropout': '0.1',
        'input_module': 'bert-base-uncased',
        'transfer_paradigm': 'finetune',
        'classifier': 'log_reg',
        'optimizer': 'bert_adam',
        'sent_enc': 'none',
        'sep_embs_for_skip': '1',
    }),
    Experiment(run_name='run12', params={
        'pretrain_tasks': '"sst,qnli"',
        'weighting_method': 'proportional',
        'dropout': '0.1',
        'input_module': 'bert-base-uncased',
        'transfer_paradigm': 'finetune',
        'classifier': 'log_reg',
        'optimizer': 'bert_adam',
        'sent_enc': 'none',
        'sep_embs_for_skip': '1',
    }),
]
