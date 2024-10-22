import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

class BatchedLatent(nn.Module):
    def __init__(self, num_adapters, rank):
        super().__init__()
        self.num_adapters = num_adapters
        self.rank = rank
        self.weight = Parameter(torch.rand(num_adapters, rank, rank))

    def forward(self, x):
        assert len(x.shape) == 3 # (batch_size, seq_length, dim)

        shape = x.shape[1:] # first dim is the expanded batch dim -> (batch_size, seq_length, dim)
        out = x.view(self.num_adapters, -1, *shape) # (b, ..) -> (num_adapters, batch_size/num_adapters, seq_length, dim)
        w = torch.transpose(self.weight, 1, 2).unsqueeze(1)
        out = torch.matmul(out, w) # -> (num_adapters, b/num_adapters, seq_length, latent_dim)
        shape = out.shape[2:]
        out = out.view(-1, *shape)
        
        return out