import torch
from torch import nn
from einops import rearrange
from tqdm import trange

from .scheduler import Scheduler
from .forwarder import Forwarder
from .optimizer import Optimizer
from .u_net import UNet
from .constants import device


class Model(nn.Module):
    def __init__(self, scheduler: Scheduler, forwarder: Forwarder, chs=(32, 64, 128)) -> None:
        super().__init__()

        self.sch = scheduler
        self.fwd = forwarder

        down_chs = [3, *chs]
        up_chs = [*chs[::-1], 1]
        self.u_net = UNet(down_chs, up_chs).to(device)

        self.time_embedding = nn.Sequential(
            nn.Linear(1, 28*28),
        ).to(device)

        self.label_embedding = nn.Sequential(
            nn.Linear(10, 28*28),
        ).to(device)

    def forward(self, x, t, label):
        b, h, w = x.shape

        t = 2 * t / (self.sch.n_steps-1) - 1  # [-1, 1]
        t = self.time_embedding(t)
        t = rearrange(t, 'b (h w) -> b 1 h w', h=h, w=w)

        label = self.label_embedding(label)
        label = rearrange(label, 'b (h w) -> b 1 h w', h=h, w=w)

        x = rearrange(x, "b h w -> b 1 h w")
        x = self.u_net(torch.cat([x, t, label], dim=1))
        x = rearrange(x, "b 1 h w -> b h w")
        return x

    def fit(self, dataloader, optimizer="adam", optimizer_kwargs={}, n_epochs=200, logger=None):
        optimizer = Optimizer(optimizer, self.parameters(), **optimizer_kwargs)

        for epoch in trange(n_epochs, desc="Epoch"):
            losses = []
            for step, (x, label) in enumerate(dataloader):
                optimizer.zero_grad()

                batch_size = x.shape[0]

                t = torch.randint(0, self.sch.n_steps, (batch_size, 1))
                t = t.to(device)
                x_noisy, noise = self.fwd.forward(x, t)

                noise_pred = self.forward(x_noisy, t, label)

                loss = torch.nn.functional.mse_loss(noise_pred, noise)
                losses.append(loss.item())

                loss.backward()
                optimizer.step()

            if epoch % (n_epochs // 10) == 0 and logger:
                logger(epoch, losses)
