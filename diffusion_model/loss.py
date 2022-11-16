import torch.nn as nn
from functools import partial


class LossGetter:
    valid_choices = [
        "mu-tild-l2",
        "mu-tild-l1",
        "x-prev-l2",
        "x-prev-l1",
        "epsilon-l2",
        "epsilon-l1",
    ]
    default = "mu-tild-l2"

    @staticmethod
    def get_loss(loss: str):
        return Loss(loss)


class Loss:
    def __init__(self, loss):
        self.loss = loss

    def is_predicting_noise(self):
        return self.loss in ["epsilon-l2", "epsilon-l1"]

    def get_mu_tilde(self, sch, t, x_noisy, noise):
        mu_tilde = 1/sch.sqrt_alpha[t] * (x_noisy - noise *
                                          (1-sch.alphas[t])/sch.sqrt_one_minus_alphas_bar[t])
        return mu_tilde

    def get_x_without_noise(self, sch, t, x_noisy, noise):
        x = 1/sch.sqrt_alpha[t] * (x_noisy - noise *
                                   (1-sch.alphas[t])/sch.sqrt_one_minus_alphas[t])
        return x

    def __call__(self, sch):
        def loss_fn(t, x_noisy, noise, model_pred):
            if self.loss == "mu-tild-l2":
                mu_tilde = self.get_mu_tilde(sch, t, x_noisy, noise)
                return nn.functional.mse_loss(mu_tilde, model_pred)
            elif self.loss == "mu-tild-l1":
                mu_tilde = self.get_mu_tilde(sch, t, x_noisy, noise)
                return nn.functional.l1_loss(mu_tilde, model_pred)
            elif self.loss == "x-prev-l2":
                x = self.get_x_without_noise(sch, t, x_noisy, noise)
                return nn.functional.mse_loss(x, model_pred)
            elif self.loss == "x-prev-l1":
                x = self.get_x_without_noise(sch, t, x_noisy, noise)
                return nn.functional.l1_loss(x, model_pred)
            elif self.loss == "epsilon-l2":
                return nn.functional.mse_loss(noise, model_pred)
            elif self.loss == "epsilon-l1":
                return nn.functional.l1_loss(noise, model_pred)
            else:
                raise ValueError(f"Unknown loss: {self.loss}")
        return loss_fn
