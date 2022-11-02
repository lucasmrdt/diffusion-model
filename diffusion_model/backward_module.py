import torch
from torch import nn
import numpy as np

from .forward_module import ForwardModule
from .scheduler import Scheduler
from .constants import device


class BackwardModule:
    def __init__(self, forward_module: ForwardModule, scheduler: Scheduler, noise_model: nn.Module, sigma_type: str = "beta"):
        self.forward_module = forward_module
        self.noise_model = noise_model
        self.scheduler = scheduler
        self.nb_steps = scheduler.nb_steps
        self.d = scheduler.input_dim
        self.sigmas = self._get_sigmas(sigma_type)

    def _get_sigmas(self, sigma_type):
        sch = self.scheduler
        if sigma_type == "beta":
            sigmas = sch.betas
        elif sigma_type == "alpha":
            sigmas = sch.betas * (1-sch.prev_alphas_bar) / (1-sch.alphas_bar)
        else:
            raise ValueError(f"Unknown sigma_type {sigma_type}")
        sigmas = sigmas.sqrt()
        return sigmas

    def batched_backward(self, x_t, t, label):
        if t > 0:
            z = torch.randn_like(x_t).to(device)
        else:
            z = torch.zeros_like(x_t).to(device)
        t = torch.full((x_t.shape[0],), t).long().to(device)
        eps_pred = self.noise_model(x_t, t, label)
        sch = self.scheduler
        return 1/(sch.sqrt_alpha[t]) * (x_t - sch.betas[t]/(sch.sqrt_one_minus_alphas_bar[t]) * eps_pred) + self.sigmas[t]*z

    def loop_backward(self, label, n_sample=1000, nb_displayed_steps=10):
        x_t = torch.randn((n_sample, *self.d)).to(device)
        xs = [x_t.detach().cpu()]
        for t in range(self.nb_steps)[::-1]:
            x_t = self.batched_backward(x_t, t, label)
            if t % (self.nb_steps // nb_displayed_steps) == 0:
                xs.append(x_t.detach().cpu())
        return xs[::-1]
