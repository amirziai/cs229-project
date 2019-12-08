# global
runs = 10
d_word = 300
lr = 0.0003
max_seq_len = 20
max_word_v_size = 100000

# per
experiments = [
    {
        'tasks': 'sst,sts-b,mrpc',
        'weighting': 'uniform',  # epsilon, ucb
        # other stuff
    }
]
