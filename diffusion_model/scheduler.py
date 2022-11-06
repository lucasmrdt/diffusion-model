import torch
from einops import rearrange

from .constants import device


def get_linear_betas(n_steps):
    ts = torch.linspace(1, 0, n_steps+1).to(device)
    betas = 1 - ts[1:]/ts[:-1]
    return betas


def get_linear_reparam_betas(n_steps):
    def f(t): return 1 - (1 - t)**2
    ts = torch.linspace(1, 0, n_steps+1).to(device)
    betas = 1 - f(ts[1:])/f(ts[:-1])
    return betas


def get_cosine_betas(n_steps):
    s = 0.0008
    def f(t): return torch.cos((t + s)/(1 + s) * torch.pi/2)**2
    ts = torch.linspace(1, 0, n_steps+1).to(device)
    betas = 1 - f(ts[1:])/f(ts[:-1])
    return betas


class Scheduler:
    def __init__(self, n_steps, schedule_type="linear-reparam", max_beta=0.999):
        self.n_steps = n_steps

        if schedule_type == "linear-reparam":
            betas = get_linear_reparam_betas(n_steps)
        elif schedule_type == "cosine":
            betas = get_cosine_betas(n_steps)
        elif schedule_type == "linear":
            betas = get_linear_betas(n_steps)
        else:
            raise ValueError(f"Unknown schedule_type {schedule_type}")

        beta0 = torch.Tensor([0]).to(device)
        betas = torch.cat([beta0, betas])
        betas = betas.clamp(0, max_beta)
        betas = rearrange(betas, "t -> t 1")

        self.betas = betas
        self.alphas = 1 - self.betas
        self.sqrt_alpha = self.alphas.sqrt()
        self.alphas_bar = self.alphas.cumprod(dim=0)
        self.alphas_bar_prev = torch.cat(
            [self.alphas_bar[:1], self.alphas_bar[:-1]])
        self.sqrt_alphas_bar = self.alphas_bar.sqrt()
        self.sqrt_one_minus_alphas_bar = (1 - self.alphas_bar).sqrt()
