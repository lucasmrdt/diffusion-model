import torch
from torch import nn
from einops import repeat

from .constants import device


class SinusoidalEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = torch.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(
            half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class LinearEmbedding(nn.Module):
    def __init__(self, dim, nb_steps):
        super().__init__()
        self.dim = dim
        self.nb_steps = nb_steps

    def forward(self, t):
        w, h = self.dim
        t = 2 * t / self.nb_steps - 1  # [-1, 1]
        t = repeat(t, 'n -> n w h', w=w, h=h)
        return t
