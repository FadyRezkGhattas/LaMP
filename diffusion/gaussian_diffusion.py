import math
from tqdm import tqdm
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .config import config

config = config['diffusion']

def get_beta_schedule(
    n_timesteps = 1000, 
    schedule = 'linear', 
    beta_start = 1e-4, 
    beta_end = 2e-2, 
    cosine_s = 8e-3, 
):  

    '''
    Returns:
        betas = (T+1)-dim np.float64 array (betas[t] = beta_t, betas[0] = dummy)
    '''

    if schedule == "linear":
        betas = np.linspace(beta_start, beta_end, n_timesteps, dtype=np.float64)
    elif schedule == "quad":
        betas = (
            np.linspace(
                beta_start**0.5,
                beta_end**0.5,
                n_timesteps,
                dtype=np.float64,
            )**2
        )
    elif schedule == "cosine":
        timesteps = (
            np.arange(n_timesteps+1, dtype=np.float64)/n_timesteps + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * math.pi/2
        alphas = np.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = np.clip(betas, min(betas), 0.999)
    elif schedule == "const":
        betas = beta_end * np.ones(n_timesteps, dtype=np.float64)
    elif schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(n_timesteps, 1, n_timesteps, dtype=np.float64)
    elif schedule == "sigmoid":
        def sigmoid(x):
            return 1 / (np.exp(-x) + 1)
        betas = np.linspace(-6, 6, n_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(schedule)

    # betas[0] as dummy
    betas = np.concatenate(([999.], betas))

    return betas

def extract(input, t, shape):

    out = torch.gather(input, 0, t)
    reshape = [shape[0]] + [1] * (len(shape)-1)
    out = out.reshape(*reshape)

    return out

class GaussianDiffusion(nn.Module):

    def __init__(self):

        super().__init__()

        self.n_timesteps = config['n_timesteps']  # T (eg, 1000)

        # compute {beta_t}_{t=1}^T
        betas = get_beta_schedule(
            n_timesteps = config['n_timesteps'],  # T
            **config['beta_schedule']
        )  # np.float64, betas[0] = dummy

        # derive {alpha_t, alpha_bar_t, beta_bar_t}_{t=1}^T
        alphas = np.concatenate(([999.], 1 - betas[1:]))  # np.float64, alphas[0] = dummy
        alphas_bar = np.concatenate(([1.], np.cumprod(alphas[1:], 0)))  # np.float64, alphas_bar[0] = 1
        betas_bar = np.concatenate(([0.], 1 - alphas_bar[1:]))  # np.float64, betas_bar[0] = 0

        # convert to torch tensors and register as members
        self.register_buffer("betas", torch.from_numpy(betas).float())
        self.register_buffer("alphas", torch.from_numpy(alphas).float())
        self.register_buffer("alphas_bar", torch.from_numpy(alphas_bar).float())
        self.register_buffer("betas_bar", torch.from_numpy(betas_bar).float())

    def q_sample(self, x_0, t, noise=None):

        '''
        Sample from q(x_t|x_0) = N(x_t; sqrt(alpha_bar_t)*x_0 + beta_bar_t*I).

        Args:
            x_0 = input (data) images; (B x C x H x W)
            t = target times; B-dim
            noise = N(0,I) samples; (B x C x H x W)

        Returns:
            x_t = samples from q(x_t|x_0); (B x C x H x W)
        '''

        if noise is None:
            noise = torch.randn_like(x_0)

        return (
            extract(self.alphas_bar.sqrt(), t, x_0.shape) * x_0
            + extract(self.betas_bar.sqrt(), t, x_0.shape) * noise
        )

    def elbo_loss_simple(self, net, x_0, t, noise=None):

        '''
        Compute the L_simple loss for the samples (x_0, t)

        Args:
            net = noise prediction network; eps(x_t, t) -> eps
            x_0 = input (data) images; (B x C x H x W)
            t = target times; B-dim
            noise = N(0,I) samples; (B x C x H x W)

        Returns:
            elbo_simple loss; scalar
        '''

        if noise is None:
            noise = torch.randn_like(x_0)  # eps

        x_noise = self.q_sample(x_0, t, noise)  # samples x_t ~ q(x_t|x_0)

        x_recon = net(x_noise, t.float())  # eps(x_t, t)

        return (noise - x_recon).square().sum(dim=1).mean(dim=0)  # E||eps(x_t, t) - eps||^2

    @torch.no_grad()
    def p_sample(self, net, taus, noise, eta, p_var):

        '''
        Skip sampling from p for a trajectory {tau_i}_{i=0}^S.

        Args:
            net = network for eps(x_t,t)
            taus = tau seqeunce; (S+1)-dim
            noise = initial samples x_{tau_S=T} ~ N(0,I); (B x C x H x W)
            eta = "eta" in sig2_t = eta * beta_tilde_t (0 for DDIM, 1 for DDPM)
            p_var = determine s2_t = sig2_t ('original'), beta_t ('ddpm_large_var')

        Returns:
            list of samples (rev-time) [x_{tau_S=T}, ..., x_{tau_1}, x_{tau_0=0}], 
                each (B x C x H x W), in cpu
        '''

        B = noise.shape[0]
        S = len(taus) - 1

        x = noise  # x_T
        traj = [noise.to('cpu'),]  # keep x_{tau_S=T}
        for i in tqdm(reversed(range(1,S+1)), total=S):  # i=S,S-1,...,1
            
            tau = (torch.ones(B) * taus[i]).to(x)  # tau_i; B-dim
            taup = (torch.ones(B) * taus[i-1]).to(x)  # tau_{i-1}; B-dim
            
            eps = net(x, tau)  # eps(x_{tau_i}, tau_i)
            
            bb = extract(self.betas_bar, tau.long(), x.shape)  # beta_bar_{tau_i}; (B x 1 x 1 x 1)
            bbp = extract(self.betas_bar, taup.long(), x.shape)  # beta_bar_{tau_{i-1}}; (B x 1 x 1 x 1)
            ab = extract(self.alphas_bar, tau.long(), x.shape)  # alpha_bar_{tau_i}; (B x 1 x 1 x 1)
            abp = extract(self.alphas_bar, taup.long(), x.shape)  # alpha_bar_{tau_{i-1}}; (B x 1 x 1 x 1)
            sig = eta * (bbp/bb*(1-ab/abp)).sqrt()  # sig_{tau_{i-1}|tau_i}(eta); (B x 1 x 1 x 1)

            mean = (abp/ab).sqrt() * (x - bb.sqrt()*eps) + (bbp - sig**2).clamp(min=0).sqrt() * eps

            if p_var == 'original':
                x = mean + sig * torch.randn_like(x)
            elif p_var == 'ddpm_large_var':
                x = mean + (1-ab/abp).sqrt() * torch.randn_like(x)
            
            traj.append(x.to('cpu'))

        return traj