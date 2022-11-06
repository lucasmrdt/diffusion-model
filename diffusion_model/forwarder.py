import torch
from tqdm import tqdm

from .constants import device
from .scheduler import Scheduler


class Forwarder:
    def __init__(self, scheduler: Scheduler):
        self.sch = scheduler

    def forward(self, x0, t):
        noise = torch.randn_like(x0).to(device)
        mean = self.sch.sqrt_alphas_bar[t] * x0
        var = self.sch.sqrt_one_minus_alphas_bar[t] * noise
        xt = mean + var
        return xt, noise

    def forward_one(self, x0, t):
        x0 = x0[None]
        xt, noise = self.forward(x0, t)
        return xt.squeeze(0), noise.squeeze(0)

    def forward_loop(self, x0, n_disp_steps=10):
        x_by_t = []
        t_space = torch.linspace(0, self.sch.n_steps, n_disp_steps-1)
        t_space = t_space.long().to(device)
        for t in tqdm(t_space, desc="Forwarding"):
            xt, _ = self.forward(x0, t)
            x_by_t.append(xt)
        return x_by_t
