import torch
import torch.nn as nn
from tqdm import tqdm

from .scheduler import Scheduler
from .constants import device
from .loss import Loss

import gc


class Backwarder:
    sigma_valid_choices = [
        "beta",
        "beta_tilde",
    ]
    sigma_default = "beta"

    def __init__(self, scheduler: Scheduler, model: nn.Module, is_predicting_noise, sigma=sigma_default):
        self.model = model
        self.sch = scheduler
        self.is_predicting_noise = is_predicting_noise
        self.sig = self._get_sigmas(sigma)

    def _get_sigmas(self, sigma):
        if sigma == "beta":
            sigmas = self.sch.betas
        elif sigma == "beta_tilde":
            sigmas = self.sch.betas * \
                (1-self.sch.alphas_bar_prev) / (1-self.sch.alphas_bar)
        else:
            raise ValueError(f"Unknown sigma {sigma}")
        sigmas = sigmas.sqrt()
        return sigmas

    @torch.no_grad()
    def backward(self, xt, t, label):
        z = (torch.randn_like(xt) if t > 1 else torch.zeros_like(xt)).to(device)
        b = xt.shape[0]
        t = torch.full((b, 1), t).to(device)
        sch = self.sch
        pred = self.model(xt, t, label)
        sig = self.sig[t]
        if self.is_predicting_noise:
            mean = 1/sch.sqrt_alpha[t] * (xt - pred*(1-sch.alphas[t]) /
                                          sch.sqrt_one_minus_alphas_bar[t])
        else:
            mean = pred
        mean = pred
        return mean + sig*z

    def backward_loop(self, label, shape, progress_bar=True):
        xt = torch.randn((shape[0], 1, *shape[1:])).to(device)
        t_space = torch.arange(self.sch.n_steps, 0, -1).long()
        if progress_bar:
            t_space = tqdm(t_space, desc="Backwarding", ascii=True)
        for t in t_space:
            with torch.no_grad():
                xt = self.backward(xt, t, label)
        return xt

    def sample(self, n_samples, shape):
        labels = torch.randint(0, 10, (n_samples,))
        x_sample = self.backward_loop(labels, (n_samples, *shape))
        return x_sample
