config = {}

config['diffusion'] = {  # hparams for forward-time model q(x_1, ... ,x_T | x_0)
    'n_timesteps': 1000,  # T
    'beta_schedule': {
        'schedule': 'linear',
        'beta_start': 1e-4,  # beta_1
        'beta_end': 2e-2,  # experiments/mini-imagenet/eval_diff/mini_imagenet_1shot_linear/final_model.ptbeta_T
    },
}

config['sampling'] = {
    # "eta" in sig2_t = eta * beta_tilde_t (0 for DDIM, 1 for DDPM, (0,1) for hybrid)
    'eta': 0,
    # s2_t = sig2_t ("original"), beta_t ("ddpm_large_var")
    'p_var': 'original',
    # skip sampling sequence length (S)
    'subseq_len': 100,
    # skip type ("quad" for cifar10, "uniform" for the rest)
    'subseq_type': 'uniform',
    # total number of rounds, ie, repetitions
    'nrounds': 10,
    # num of candidates per time step
    'cand_size': 500,
    # greedy beam search size
    'beam_size': 1,
    'reg_weight': 0
}