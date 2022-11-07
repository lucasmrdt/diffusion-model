import torch
from torch import nn
from einops import rearrange
from tqdm import tqdm

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
            nn.ReLU(),
            nn.Unflatten(1, (28, 28)),
        ).to(device)

        self.label_embedding = nn.Sequential(
            nn.Linear(10, 28*28),
            nn.ReLU(),
            nn.Unflatten(1, (28, 28)),
        ).to(device)

    def forward(self, x, t, label):
        t = 2 * t / self.sch.n_steps - 1  # [-1, 1]
        t = self.time_embedding(t)

        label = self.label_embedding(label)

        x, t, label = x[:, None], t[:, None], label[:, None]

        out = self.u_net(torch.cat([x, t, label], dim=1))
        out = out.squeeze(1)
        return out

    def forward_one(self, x, t, label):
        x, t, label = x[None], t[None], label[None]
        out = self.forward(x, t, label)
        return out.squeeze(0)

    def fit(self, dataloader, optimizer="adam", optimizer_kwargs={}, n_epochs=200, logger=None):
        optimizer = Optimizer(optimizer, self.parameters(), **optimizer_kwargs)

        for epoch in range(n_epochs):
            losses = []
            for step, (x, label) in tqdm(enumerate(dataloader), desc=f"Epoch {epoch}", total=len(dataloader)):
                optimizer.zero_grad()

                batch_size = x.shape[0]

                t = torch.randint(1, self.sch.n_steps+1, (batch_size, 1))
                t = t.to(device)

                x_noisy, noise = self.fwd.forward(x, t)

                noise_pred = self.forward(x_noisy, t, label)

                loss = torch.nn.functional.mse_loss(noise, noise_pred)
                losses.append(loss.item())

                loss.backward()
                optimizer.step()

            if epoch % 10 == 0 and logger:
                logger(epoch, losses)
