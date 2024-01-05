"""
Model submodules including reversible instance normalization, patch embedding, and transformer block.
"""
import math
import torch
import torch.nn as nn
from torch.nn import functional as F


import torch
import torch.nn as nn


class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False):
        super().__init__()
        self.dims, self.contiguous = dims, contiguous

    def forward(self, x):
        if self.contiguous:
            return x.transpose(*self.dims).contiguous()
        else:
            return x.transpose(*self.dims)


class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        """
        Args:
            num_features: the number of features or channels
            eps: a value added for numerical stability
            affine: if True, RevIN has learnable affine parameters
        """
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self._init_params()

    def forward(self, x: torch.Tensor, mode: str):
        if mode == "denorm":
            x = self._denormalize(x)
        elif mode == "norm":
            self._get_statistics(x)
            x = self._normalize(x)
        else:
            raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x: torch.Tensor):
        """
        x: (B, L, N)
        """
        dim2reduce = tuple(range(1, x.ndim - 1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(
            torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps
        ).detach()

    def _normalize(self, x: torch.Tensor):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x: torch.Tensor):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        return x + self.mean


class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """

    def forward(self, x):
        return (
            0.5
            * x
            * (
                1.0
                + torch.tanh(
                    math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))
                )
            )
        )


class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(
        self,
        block_size: int,
        n_embd: int,
        n_head: int,
        attn_pdrop: float,
        resid_pdrop: float,
    ):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        # output projection
        self.c_proj = nn.Linear(n_embd, n_embd)
        # regularization
        self.attn_dropout = nn.Dropout(attn_pdrop)
        self.resid_dropout = nn.Dropout(resid_pdrop)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(block_size, block_size)).view(
                1, 1, block_size, block_size
            ),
        )
        self.n_head = n_head
        self.n_embd = n_embd

    def forward(self, x):
        (
            B,
            T,
            C,
        ) = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        # re-assemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class Block(nn.Module):
    norm_types = ["batch", "layer"]

    def __init__(
        self,
        block_size: int,
        n_embd: int,
        n_head: int,
        attn_pdrop: float,
        resid_pdrop: float,
        norm: str = "batch",
    ):
        super().__init__()
        assert norm in self.norm_types, f"norm type must be one of {self.norm_types}"
        if norm == "batch":
            self.norm_attn = nn.Sequential(
                Transpose(1, 2), nn.BatchNorm1d(n_embd), Transpose(1, 2)
            )
        else:
            self.norm_attn = nn.LayerNorm(n_embd)

        self.attn = CausalSelfAttention(
            block_size=block_size,
            n_embd=n_embd,
            n_head=n_head,
            attn_pdrop=attn_pdrop,
            resid_pdrop=resid_pdrop,
        )

        if norm == "batch":
            self.norm_ffn = nn.Sequential(
                Transpose(1, 2), nn.BatchNorm1d(n_embd), Transpose(1, 2)
            )
        else:
            self.norm_ffn = nn.LayerNorm(n_embd)

        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = nn.ModuleDict(
            dict(
                c_fc=nn.Linear(n_embd, 4 * n_embd),
                c_proj=nn.Linear(4 * n_embd, n_embd),
                act=NewGELU(),
                dropout=nn.Dropout(resid_pdrop),
            )
        )
        m = self.mlp
        self.mlpf = lambda x: m.dropout(m.c_proj(m.act(m.c_fc(x))))  # MLP forward

    def forward(self, x):
        x = x + self.attn(self.norm_attn(x))
        x = x + self.mlpf(self.norm_ffn(x))
        return x


class FlattenHead(nn.Module):
    def __init__(self, individual, n_vars, nf, target_window, head_dropout=0.1):
        super().__init__()

        self.individual = individual
        self.n_vars = n_vars

        if self.individual:
            self.linears = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            self.flattens = nn.ModuleList()
            for _ in range(self.n_vars):
                self.flattens.append(nn.Flatten(start_dim=-2))
                self.linears.append(nn.Linear(nf, target_window))
                self.dropouts.append(nn.Dropout(head_dropout))
        else:
            self.flatten = nn.Flatten(start_dim=-2)
            self.linear = nn.Linear(nf, target_window)
            self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        """
        x: B, N, E, P
        """
        if self.individual:
            x_out = []
            for i in range(self.n_vars):
                z = self.flattens[i](x[:, i, :, :])  # B, E * P
                z = self.linears[i](z)  # B, Y
                z = self.dropouts[i](z)
                x_out.append(z)
            x = torch.stack(x_out, dim=1)  # B, N, Y
        else:
            x = self.flatten(x)
            x = self.linear(x)
            x = self.dropout(x)
        return x


class PoolingHead(nn.Module):
    def __init__(self, n_embd, target_window, head_dropout=0.1):
        super().__init__()

        self.linear = nn.Linear(n_embd, target_window)

        self.dropout = nn.Dropout(head_dropout)

        self.pooling = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        """
        x: B, N, E, P
        """
        x = self.pooling(x).squeeze(-1)  # B, N, E

        x = self.linear(x)  # B, N, Y

        x = self.dropout(x)  # B, N, Y

        return x
