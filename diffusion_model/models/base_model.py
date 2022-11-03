import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
from einops import rearrange

from ..optimizers import get_optimizer
from ..constants import device


def get_optimizer(optimizer):
    if optimizer == 'adam':
        return torch.optim.Adam
    elif optimizer == 'sgd':
        return torch.optim.SGD
    else:
        raise ValueError('Unknown optimizer')


class BaseNoiseModel(nn.Module):
    def __init__(self, forward_module, scheduler):
        super().__init__()
        self.forward_module = forward_module
        self.scheduler = scheduler
        self.nb_steps = scheduler.nb_steps

    def forward(self, x, t, label):
        raise NotImplementedError

    def fit(self, dataloader, optimizer="adam", optimizer_kwargs={}, nb_epochs=200, logger=None):
        optimizer = get_optimizer(optimizer)
        optimizer = optimizer(self.parameters(), **optimizer_kwargs)

        for epoch in tqdm(range(nb_epochs)):
            losses = []
            for step, (x, label) in enumerate(dataloader):
                optimizer.zero_grad()

                batch_size = x.shape[0]

                t = torch.randint(0, self.nb_steps, (batch_size,)).to(device)
                batch_noisy, noise = self.forward_module.batched_forward(x, t)

                label = rearrange(label, "b -> b ()")
                t = rearrange(t, "b -> b ()")
                batch_noisy = rearrange(batch_noisy, "b h w -> b () h w")
                noise_pred = self.forward(batch_noisy, t, label)
                noise_pred = rearrange(noise_pred, "b 1 h w -> b h w")

                loss = F.mse_loss(noise_pred, noise)
                losses.append(loss.item())

                loss.backward()
                optimizer.step()

            if epoch % (nb_epochs // 10) == 0:
                if logger:
                    logger(epoch, losses)
