import torch
from einops import repeat

from .constants import device


def get_linear_reparam_betas(nb_steps, min_alpha_bar=0, max_alpha_bar=1, max_beta=0.999):
    def f(t): return 1 - (1 - t)**2
    ts = torch.linspace(max_alpha_bar, min_alpha_bar, nb_steps+1).to(device)
    betas = 1 - f(ts[1:])/f(ts[:-1])
    betas = betas.clamp(0, max_beta)
    return betas


def get_cosine_betas(nb_steps, min_alpha_bar=0, max_alpha_bar=1, max_beta=0.999):
    s = 0.0008
    def f(t): return torch.cos((t + s)/(1 + s) * torch.pi/2)**2
    ts = torch.linspace(min_alpha_bar, max_alpha_bar, nb_steps+1).to(device)
    betas = 1 - f(ts[1:])/f(ts[:-1])
    betas = betas.clamp(0, max_beta)
    return betas


class Scheduler:
    def __init__(self, input_dim, nb_steps, scheduler="linear-reparam", scheduler_kwargs={}):
        self.nb_steps = nb_steps

        if isinstance(input_dim, int):
            input_dim = (input_dim,)
        self.input_dim = input_dim

        if scheduler == "linear-reparam":
            betas = get_linear_reparam_betas(nb_steps, **scheduler_kwargs)
        elif scheduler == "cosine":
            betas = get_cosine_betas(nb_steps, **scheduler_kwargs)
        else:
            raise ValueError(f"Unknown scheduler {scheduler}")

        ds = {f"d{i}": d for i, d in enumerate(input_dim)}
        self.betas = repeat(betas, f"t -> t {' '.join(ds.keys())}", **ds)

        self.alphas = 1 - self.betas
        self.sqrt_alpha = self.alphas.sqrt()
        self.alphas_bar = self.alphas.cumprod(dim=0)
        self.alphas_bar_prev = torch.cat(
            [torch.ones(1, *input_dim).to(device), self.alphas_bar[:-1]])
        self.sqrt_alphas_bar = self.alphas_bar.sqrt()
        self.sqrt_one_minus_alphas_bar = (1 - self.alphas_bar).sqrt()
