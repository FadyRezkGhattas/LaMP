import math

import torch
import torch.nn as nn

def get_timestep_embedding(timesteps, embedding_dim):

    '''
    Build sinusoidal embeddings.
    This matches the implementation in DDPM from fairseq.
    This matches the implementation in tensor2tensor, but differs slightly
      from the description in Section 3.5 of "Attention Is All You Need".
    '''

    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    
    return emb

def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)

class AdditiveEmbedMLP(nn.Module):
    def __init__(self, indim=739, hdim=2048, nhids=3):
        
        super().__init__()

        self.nhids = nhids
        self.hdim = hdim
        self.temb_hdim = hdim*4
        self.indim = indim

        # timestep embedding
        dims = [indim,] + [hdim for _ in range(nhids)]
        self.temb = nn.Module()
        self.temb.dense = nn.ModuleList([
            torch.nn.Linear(hdim, self.temb_hdim),
            torch.nn.Linear(self.temb_hdim, hdim),
        ])
        self.temb_hidden_layers = nn.ModuleList([
            torch.nn.Linear(dims[i+1], dims[i]) for i in range(len(dims)-1)
        ])

        # fully connected layers
        nonlins = [nn.ReLU() for _ in range(nhids)]
        dims = [indim,] + [hdim for _ in range(nhids)]
        self.layers = torch.nn.Sequential()
        for i in range(len(dims)-1):
            self.layers.add_module(
                name = 'layer{0:d}'.format(i),
                module = torch.nn.Sequential(
                    torch.nn.Linear(dims[i], dims[i+1]), 
                    nonlins[i],
                    nn.LayerNorm([dims[i+1]])
                )
            )
        self.layers.add_module(
            name = 'head',
            module = torch.nn.Linear(dims[-1], indim), 
        )

    def forward(self, x, t):
        # (MY modification) since we assume t=1-base (instead of t=0-base), we subtract 1 from t for compatibility
        t = t - 1.0

        # timestep embedding
        temb = get_timestep_embedding(t, self.hdim)  # (B x hdim)
        temb = self.temb.dense[0](temb)
        temb = nonlinearity(temb)
        temb = self.temb.dense[1](temb)  # (B x 4*hdim)

        temb_h = self.temb_hidden_layers[0](temb)
        h = self.layers[0](x+temb_h)
        for i in range(1, self.nhids):
            temb_h = self.temb_hidden_layers[i](temb)
            h = self.layers[i](h+temb_h)
        h = self.layers[-1](h+temb)

        return h