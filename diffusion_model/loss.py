import torch.nn as nn

from .scheduler import Scheduler


class Loss:
    valid_choices = [
        "mu-tild-l2",
        "mu-tild-l1",
        "x-prev-l2",
        "x-prev-l1",
        "epsilon-l2",
        "epsilon-l1",
    ]
    default = "mu-tild-l2"

    def __init__(self, scheduler: Scheduler, loss: str):
        self.sch = scheduler
        self.loss = loss

    def get_mu_tilde(self, t, x_noisy, noise):
        sch = self.sch
        mu_tilde = 1/sch.sqrt_alpha[t] * (x_noisy - noise *
                                          (1-sch.alphas[t])/sch.sqrt_one_minus_alphas_bar[t])
        return mu_tilde

    def get_x_without_noise(self, t, x_noisy, noise):
        sch = self.sch
        x = 1/sch.sqrt_alpha[t] * (x_noisy - noise *
                                   (1-sch.alphas[t])/sch.sqrt_one_minus_alphas[t])
        return x

    def __call__(self, t, x_noisy, noise, model_pred):
        if self.loss == "mu-tild-l2":
            mu_tilde = self.get_mu_tilde(t, x_noisy, noise)
            return nn.functional.mse_loss(mu_tilde, model_pred)
        elif self.loss == "mu-tild-l1":
            mu_tilde = self.get_mu_tilde(t, x_noisy, noise)
            return nn.functional.l1_loss(mu_tilde, model_pred)
        elif self.loss == "x-prev-l2":
            x = self.get_x_without_noise(t, x_noisy, noise, model_pred)
            return nn.functional.mse_loss(x, model_pred)
        elif self.loss == "x-prev-l1":
            x = self.get_x_without_noise(t, x_noisy, noise, model_pred)
            return nn.functional.l1_loss(x, model_pred)
        elif self.loss == "epsilon-l2":
            return nn.functional.mse_loss(noise, model_pred)
        elif self.loss == "epsilon-l1":
            return nn.functional.l1_loss(noise, model_pred)
        else:
            raise ValueError(f"Unknown loss: {self.loss}")
