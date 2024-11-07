from tqdm import tqdm

import torch
import numpy as np

from .gaussian_diffusion import extract
from .config import config

sampling_config = config['sampling']


def greedy_sample(dm, net, rounds=10):
    params_vectors = []
    for round in range(rounds):
        params_vectors_ = greedy_sample_round(dm, net)
        params_vectors += params_vectors_
    return params_vectors

def greedy_sample_round(dm, net):
    '''
    Diffusion guided greedy sampling.
    '''
    in_dim = net.indim
    device = 'cuda'
    config = sampling_config
    eta, p_var = config['eta'], config['p_var']
    net.eval()
    
    # subseq {tau_i}_{i=0}^S
    T = dm.n_timesteps
    S = config['subseq_len']
    if config['subseq_type'] == "uniform":
        taus = np.linspace(1, T, S)  # tau_i = c*i; make sure tau_1=1, tau_S=T
        taus = np.concatenate(([0,], taus))  # prepend tau_0 = 0
        taus = [int(s) for s in list(taus)]
        taus = np.unique(taus).tolist()  # (MY) remove redundancy if any
    elif config['subseq_type'] == "quad":
        taus = np.linspace(1, np.sqrt(T*0.8), S-1) ** 2  # tau_i = c*(i**2); make sure tau_1=1
        taus = np.concatenate(([0,], taus))  # prepend tau_0 = 0
        taus = np.concatenate((taus, [T,]))  # append tau_S = T
        taus = [int(s) for s in list(taus)]
        taus = np.unique(taus).tolist()  # (MY) remove redundancy if any
    else:
        raise NotImplementedError
        
    with torch.no_grad():
        
        noise = torch.randn(
            sampling_config['cand_size'], 
            in_dim
        ).to(device)  # x_{tau_S=T}; (#cands x dim(head))
        B = noise.shape[0]
        adapters = noise  # x_{tau_S=T}
        for i in tqdm(reversed(range(1,S+1)), total=S):  # i=S,S-1,...,1
            
            tau = (torch.ones(B) * taus[i]).to(device)  # tau_i; B-dim
            taup = (torch.ones(B) * taus[i-1]).to(device)  # tau_{i-1}; B-dim
            
            eps = net(adapters, tau)  # eps(x_{tau_i}, tau_i)
            
            bb = extract(dm.betas_bar, tau.long(), adapters.shape)  # beta_bar_{tau_i}
            bbp = extract(dm.betas_bar, taup.long(), adapters.shape)  # beta_bar_{tau_{i-1}}
            ab = extract(dm.alphas_bar, tau.long(), adapters.shape)  # alpha_bar_{tau_i}
            abp = extract(dm.alphas_bar, taup.long(), adapters.shape)  # alpha_bar_{tau_{i-1}}
            sig = eta * (bbp/bb*(1-ab/abp)).sqrt()  # sig_{tau_{i-1}|tau_i}(eta)
            mean = (abp/ab).sqrt() * (adapters - bb.sqrt()*eps) + (bbp - sig**2).clamp(min=0).sqrt() * eps
            if p_var == 'original':
                adapters = mean + sig * torch.randn_like(adapters)
            elif p_var == 'ddpm_large_var':
                adapters = mean + (1-ab/abp).sqrt() * torch.randn_like(adapters)
    
    return adapters

def posterior_sample(dm, net, model, get_loss_grads, support_batch, cand_size, timestep_dps):
    '''
    Diffusion guided greedy sampling.
    '''
    in_dim = net.indim
    device = 'cuda'
    config = sampling_config
    eta, p_var = config['eta'], config['p_var']
    net.eval()
    
    # subseq {tau_i}_{i=0}^S
    T = dm.n_timesteps
    S = config['subseq_len']
    if config['subseq_type'] == "uniform":
        taus = np.linspace(1, T, S)  # tau_i = c*i; make sure tau_1=1, tau_S=T
        taus = np.concatenate(([0,], taus))  # prepend tau_0 = 0
        taus = [int(s) for s in list(taus)]
        taus = np.unique(taus).tolist()  # (MY) remove redundancy if any
    elif config['subseq_type'] == "quad":
        taus = np.linspace(1, np.sqrt(T*0.8), S-1) ** 2  # tau_i = c*(i**2); make sure tau_1=1
        taus = np.concatenate(([0,], taus))  # prepend tau_0 = 0
        taus = np.concatenate((taus, [T,]))  # append tau_S = T
        taus = [int(s) for s in list(taus)]
        taus = np.unique(taus).tolist()  # (MY) remove redundancy if any
    else:
        raise NotImplementedError
        
    noise = torch.randn(
        cand_size, 
        in_dim
    ).to(device)  # x_{tau_S=T}; (#cands x dim(head))
    B = noise.shape[0]
    adapters = noise  # x_{tau_S=T}
    for i in tqdm(reversed(range(1,S+1)), total=S, desc='Timesteps', position=0, leave=True, ncols=80):  # i=S,S-1,...,1
        
        tau = (torch.ones(B) * taus[i]).to(device)  # tau_i; B-dim
        taup = (torch.ones(B) * taus[i-1]).to(device)  # tau_{i-1}; B-dim
        
        eps = net(adapters, tau)  # eps(x_{tau_i}, tau_i)
        
        bb = extract(dm.betas_bar, tau.long(), adapters.shape)  # beta_bar_{tau_i}
        bbp = extract(dm.betas_bar, taup.long(), adapters.shape)  # beta_bar_{tau_{i-1}}
        ab = extract(dm.alphas_bar, tau.long(), adapters.shape)  # alpha_bar_{tau_i}
        abp = extract(dm.alphas_bar, taup.long(), adapters.shape)  # alpha_bar_{tau_{i-1}}
        sig = eta * (bbp/bb*(1-ab/abp)).sqrt()  # sig_{tau_{i-1}|tau_i}(eta)
        mean = (abp/ab).sqrt() * (adapters - bb.sqrt()*eps) + (bbp - sig**2).clamp(min=0).sqrt() * eps
        # dps
        losses = []
        if i < timestep_dps:
            alpha = ab / abp
            beta = 1. - alpha
            x0_tweedie = (adapters - bb.sqrt()*eps) / ab.sqrt()
            means_temp = []
            for j, adapter in tqdm(enumerate(x0_tweedie), total=len(x0_tweedie), desc='DPS Adapters', position=1, leave=False, ncols=80):
                loss, grad = get_loss_grads(model, adapter, support_batch)
                losses.append(loss)
                mean_temp = mean[j] + beta[j] * grad
                means_temp.append(mean_temp)
            mean = torch.vstack(means_temp).to(device)
        if p_var == 'original':
            adapters = mean + sig * torch.randn_like(adapters)
        elif p_var == 'ddpm_large_var':
            adapters = mean + (1-ab/abp).sqrt() * torch.randn_like(adapters)
    
    return adapters, losses