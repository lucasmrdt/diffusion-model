import torch
from einops import repeat, rearrange

from .constants import device


class ForwardModule:
    def __init__(self, scheduler):
        self.scheduler = scheduler
        self.nb_steps = scheduler.nb_steps

    def batched_forward(self, xs_0, time_steps):
        sch = self.scheduler
        noise = torch.randn_like(xs_0)
        sqrt_alphas_bar_t = sch.sqrt_alphas_bar[time_steps]
        sqrt_one_minus_alphas_bar_t = sch.sqrt_one_minus_alphas_bar[time_steps]
        x_t = sqrt_alphas_bar_t * xs_0 + sqrt_one_minus_alphas_bar_t * noise
        return x_t, noise

    def loop_forward(self, x_0, nb_displayed_steps=10):
        n_samples = x_0.shape[0]
        t_step = self.nb_steps // nb_displayed_steps
        t = torch.arange(0, self.nb_steps, step=t_step).to(device)
        ts = repeat(t, "t -> (n t)", n=n_samples)
        xs_0 = repeat(x_0, "n d -> (n t) d", t=nb_displayed_steps)
        xs_t, _ = self.batched_forward(xs_0, ts)
        xs_t = rearrange(xs_t, "(n t) d -> t n d", t=nb_displayed_steps)
        return xs_t
