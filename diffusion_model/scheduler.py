import torch

from .constants import device


def get_betas_from_gammas_bar_f(n_steps, f):
    ts = torch.linspace(1, 0, n_steps+1).to(device)
    betas = 1 - f(ts[1:])/f(ts[:-1])
    return betas


def get_betas_with_linear_noise(n_steps: int):
    def f(t): return 1 - (1 - t)**2
    return get_betas_from_gammas_bar_f(n_steps, f)


def get_betas_with_linear_x(n_steps: int):
    def f(t): return t**2
    return get_betas_from_gammas_bar_f(n_steps, f)


def get_betas_with_cosine(n_steps: int):
    s = 0.0008
    def f(t): return torch.cos((t + s)/(1 + s) * torch.pi/2)**2
    return get_betas_from_gammas_bar_f(n_steps, f)


def get_betas_with_linear_gamma_bar(n_steps: int):
    return get_betas_from_gammas_bar_f(n_steps, lambda t: t)


class Scheduler:
    valid_choices = ["linear-gamma-bar", "linear-noise", "linear-x", "cosine"]
    default = "linear-noise"

    def __init__(self, n_steps: int, schedule_type: str, max_beta=0.999):
        self.n_steps = n_steps

        if schedule_type == "linear-noise":
            betas = get_betas_with_linear_noise(n_steps)
        elif schedule_type == "linear-x":
            betas = get_betas_with_linear_x(n_steps)
        elif schedule_type == "cosine":
            betas = get_betas_with_cosine(n_steps)
        elif schedule_type == "linear-gamma-bar":
            betas = get_betas_with_linear_gamma_bar(n_steps)
        else:
            raise ValueError(f"Unknown schedule_type {schedule_type}")

        beta0 = torch.Tensor([0]).to(device)
        betas = torch.cat([beta0, betas])
        betas = betas.clamp(0, max_beta)
        # transform to (n_steps+1) x 1 x 1 x 1 (channel, height, width)
        betas = betas[:, None, None]

        self.betas = betas
        self.alphas = 1 - self.betas
        self.sqrt_alpha = self.alphas.sqrt()
        self.sqrt_one_minus_alphas = (1 - self.alphas).sqrt()
        self.alphas_bar = self.alphas.cumprod(dim=0)
        self.alphas_bar_prev = torch.cat(
            [self.alphas_bar[:1], self.alphas_bar[:-1]])
        self.sqrt_alphas_bar = self.alphas_bar.sqrt()
        self.sqrt_one_minus_alphas_bar = (1 - self.alphas_bar).sqrt()
