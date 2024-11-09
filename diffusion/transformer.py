# adapted from ProtoDiff
import math

import numpy as np
import torch
import torch.nn as nn

from copy import deepcopy
from torch.nn import functional as F


class SelfAttention(nn.Module):
    """
    A vanilla multi-head self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, n_embd, n_head, attn_pdrop=0.0, resid_pdrop=0.0):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        # output projection
        self.proj = nn.Linear(n_embd, n_embd)

        self.n_head = n_head

    def forward(self, x):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y


class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, n_embd, n_head, attn_pdrop=0.0, resid_pdrop=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = SelfAttention(n_embd, n_head, attn_pdrop, resid_pdrop)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(resid_pdrop),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class DecoderStack(nn.Module):
    def __init__(
        self, context_window, n_layer=12, n_head=12, token_hidden_dim_size=768, attn_pdrop=0.0,
        resid_pdrop=0.0, embd_pdrop=0.0,
    ):
        """
        Args:
        n_layer (int, optional): Number of transformer blocks to use. Defaults to 12.
        n_head (int, optional): Number of attention heads in the transformer. Defaults to 12.
        token_hidden_dim_size (int, optional): Number of embedding dimensions. Defaults to 768.
        attn_pdrop (float, optional): Probability of dropping out the attention weights in the transformer. Defaults to 0.0.
        resid_pdrop (float, optional): Probability of dropping out the residual connection in the transformer. Defaults to 0.0.
        embd_pdrop (float, optional): Probability of dropping out the input embeddings. Defaults to 0.0.
        """
        super().__init__()
        self.n_tokens = context_window

        # transformer
        self.blocks = nn.Sequential(
            *[Block(token_hidden_dim_size, n_head, attn_pdrop, resid_pdrop) for _ in range(n_layer)]
        )
        
        # self.build_encoder(n_embd, encoder_depth, self.num_parameters)
        self.ln_in = nn.LayerNorm(token_hidden_dim_size)
        self.apply(self._init_weights)

    @staticmethod
    def build_decoder(n_embd, decoder_depth, output_splits):
        # Create a unique MLP decoder for each noised token
        output_parameter_projections = nn.ModuleList()
        for output_chunk_size in output_splits:
            out_proj = []
            for _ in range(decoder_depth - 1):
                out_proj.append(nn.Linear(n_embd, n_embd, bias=False))
                out_proj.append(nn.GELU())
            out_proj.append(nn.Linear(n_embd, output_chunk_size, bias=False))
            out_proj = nn.Sequential(*out_proj)
            output_parameter_projections.append(out_proj)
        return output_parameter_projections

    def get_module_size(self):
        return self.module_size

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, (nn.Linear, nn.Conv1d)) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x):
        x = self.blocks(x)
        return x


class FrequencyEmbedder(nn.Module):

    def __init__(self, num_frequencies, max_freq_log2):
        super().__init__()
        frequencies = 2 ** torch.linspace(0, max_freq_log2, steps=num_frequencies)
        self.register_buffer('frequencies', frequencies)

    def forward(self, x):
        # x should be of size (N,) or (N, D)
        N = x.size(0)
        if x.dim() == 1:  # (N,)
            x = x.unsqueeze(1)  # (N, D) where D=1
        x_unsqueezed = x.unsqueeze(-1).to('cuda', torch.float)  # (N, D, 1)
        scaled = self.frequencies.view(1, 1, -1) * x_unsqueezed  # (N, D, num_frequencies)
        s = torch.sin(scaled)
        c = torch.cos(scaled)
        #embedded = torch.cat([s, c], dim=-1).view(N, -1)  # (N, D * 2 * num_frequencies + D)
        embedded = torch.cat([s, c, x_unsqueezed], dim=-1).view(N, -1)
        return embedded


class GPT(nn.Module):
    """
    The GPT model.
    """

    def __init__(
        self,
        config,
        n_layer=12,
        n_head=12,
        token_hidden_dim_size=768,
        attn_pdrop=0.0,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        num_frequencies=256,
        max_freq_log2=20,
    ):
        '''
        args:
        - config (dict) : the configuration of the model.
            - indim (int) : the number of parameters to be embedded.
            - module_size (int) : the size of the modules (each module is a subset of parameters that are treated as one token. In lora-xs for example, it is the 6x6 square matrix).
        - n_layer (int, optional): Number of transformer blocks to use. Defaults to 12.
        - n_head (int, optional): Number of attention heads in the transformer. Defaults to 12.
        - token_hidden_dim_size (int, optional): Number of embedding dimensions. Defaults to 768.
        - attn_pdrop (float, optional): Probability of dropping out the attention weights in the transformer. Defaults to 0.0.
        - resid_pdrop (float, optional): Probability of dropping out the residual connection in the transformer. Defaults to 0.0.
        - embd_pdrop (float, optional): Probability of dropping out the input embeddings. Defaults to 0.0.
        - num_frequencies (int) : the number of frequencies sampled for embedding scalars.
        - max_freq_log2 (int) : the max log2 frequency for embedding scalars.
        '''
        super().__init__()
        assert 0 == config['indim'] % (config['module_size']), "number of parameters must be divisible by block size"
        self.num_parameters, self.module_size = config['indim'], config['module_size']
        self.n_tokens = (self.num_parameters // self.module_size) + 1 # +1 for the time embedding token
        self.time_embedding_size = self.get_scalar_token_size(num_frequencies)

        # frequency embedder for the time token constant
        self.scalar_embedder = FrequencyEmbedder(num_frequencies, max_freq_log2)

        # project each module and the time token embedding into token_hidden_dim
        self.module_projection = nn.Linear(self.module_size, token_hidden_dim_size)
        self.time_embed_projection = nn.Linear(self.time_embedding_size, token_hidden_dim_size)

        # input embedding stem
        self.pos_emb = nn.Parameter(torch.zeros(1, self.n_tokens, token_hidden_dim_size))
        self.drop = nn.Dropout(embd_pdrop)

        self.decoder = DecoderStack(self.n_tokens, n_layer, n_head, token_hidden_dim_size, attn_pdrop, resid_pdrop, embd_pdrop)

        self.output_layer = nn.Linear(token_hidden_dim_size, self.module_size)

        self.apply(self.decoder._init_weights)

    @staticmethod
    def get_scalar_token_size(num_frequencies):
        """
        Computes the size of each metadata token after being projected by the frequency embedder.
        """
        return num_frequencies * 2 + 1

    def encode_parameters(self, x, time_embedding):
        """
        Chunk input parameter vector, apply per-chunk encoding, and
        stack projected chunks along the sequence (token) dimension.
        """
        assert len(x.shape) == 2, f"Expected 2D input (batch size x number of parameters), got {len(x.shape)}D."
        x = x.view(x.size(0), self.module_size, -1).transpose(1,2)   # (b, num modules, module_size)
        tokenized_modules = self.module_projection(x) # (b, num modules, h)
        time_embed_token = self.time_embed_projection(time_embedding) # (b, h)
        # append time embedding token to tokenized modules
        tokens = torch.cat((tokenized_modules, time_embed_token.unsqueeze(1)), dim=1)
        return tokens
    
    def forward(self, x, t):
        t = t.long()
        t_embedding = self.scalar_embedder(t)

        embeddings = self.encode_parameters(x, t_embedding)
        b, t, d = embeddings.size()
        assert t == self.n_tokens, f"Expected {self.n_tokens} tokens on dim=1, but got {t}"
        # forward the GPT model
        x = self.drop(embeddings + self.pos_emb)

        h = self.decoder(x)
        output = self.output_layer(h[:, :-1, :]).view(b, -1)
        return output