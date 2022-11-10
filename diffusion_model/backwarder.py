import torch
from tqdm import tqdm
from einops import rearrange

from .scheduler import Scheduler
from .constants import device
from .model import Model


class Backwarder:
    def __init__(self, scheduler: Scheduler, model: Model, sigma="beta"):
        self.model = model
        self.sch = scheduler
        self.sig = self._get_sigmas(sigma)

    def _get_sigmas(self, sigma):
        if sigma == "beta":
            sigmas = self.sch.betas
        elif sigma == "alpha":
            sigmas = self.sch.betas * \
                (1-self.sch.alphas_bar_prev) / (1-self.sch.alphas_bar)
        else:
            raise ValueError(f"Unknown sigma {sigma}")
        sigmas = sigmas.sqrt()
        return sigmas

    def backward(self, xt, t, label):
        z = (torch.randn_like(xt) if t > 1 else torch.zeros_like(xt)).to(device)
        b = xt.shape[0]
        t = torch.full((b, 1), t).to(device)
        mean = self.model(xt, t, label)
        # return 1/self.sch.sqrt_alpha[t] * (xt - noise_pred*(1 - self.sch.alphas[t])/self.sch.sqrt_one_minus_alphas_bar[t]) + self.sig[t]*z
        return mean + self.sig[t]*z
        # return mean

    def backward_loop(self, label, shape, n_disp_steps=10, progress_bar=True):
        xt = torch.randn(shape).to(device)
        # t_space = torch.linspace(self.sch.n_steps, 1, n_disp_steps-1).long()
        t_space = torch.arange(self.sch.n_steps, 0, -1).long()
        if progress_bar:
          t_space = tqdm(t_space, desc="Backwarding")

        x_by_t = [xt.detach().cpu()]
        for t in t_space:
            xt = self.backward(xt, t, label)
            x_by_t.append(xt.detach().cpu())
        return x_by_t