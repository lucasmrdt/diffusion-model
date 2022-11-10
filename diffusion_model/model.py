import torch
from torch import nn
from tqdm import tqdm

from .scheduler import Scheduler
from .forwarder import Forwarder
from .optimizer import Optimizer
from .loss import Loss
from .u_net import UNet
from .constants import device


class Model(nn.Module):
    def __init__(self, scheduler: Scheduler, forwarder: Forwarder, chs=(32, 64, 128)) -> None:
        super().__init__()

        self.sch = scheduler
        self.fwd = forwarder

        down_chs = [3, *chs]
        up_chs = [*chs[::-1], 1]
        self.u_net = UNet(down_chs, up_chs)

        self.time_embedding = nn.Sequential(
            nn.Linear(1, 32*32),
            nn.ReLU(),
            nn.Unflatten(1, (32, 32)),
        )

        self.label_embedding = nn.Sequential(
            nn.Linear(10, 32*32),
            nn.ReLU(),
            nn.Unflatten(1, (32, 32)),
        )

    def forward(self, X, t, label):
        t = 2 * t / self.sch.n_steps - 1  # [-1, 1]
        t = self.time_embedding(t)
        t = t[:, None]

        label = self.label_embedding(label)
        label = label[:, None]

        input = torch.cat([X, label, t], dim=1)
        out = self.u_net(input)
        return out

    def _one_step(self, loss_fn: Loss, X, label):
        batch_size = X.shape[0]
        t = torch.randint(1, self.sch.n_steps+1, (batch_size, 1))

        label = nn.functional.one_hot(label, 10).float()

        X, label, t = X.to(device), label.to(device), t.to(device)

        x_noisy, noise = self.fwd.forward(X, t)
        model_pred = self.forward(x_noisy, t, label)

        return loss_fn(t, x_noisy, noise, model_pred)

    def one_step_eval(self, loss_fn: Loss, X, label):
        loss = self._one_step(loss_fn, X, label)
        return loss.item()

    def one_step_training(self, optimizer: torch.optim.Optimizer, loss_fn: Loss, X, label):
        optimizer.zero_grad()

        loss = self._one_step(loss_fn, X, label)
        loss.backward()
        optimizer.step()

        return loss.item()

    def fit(self, dataloader, optimizer="adam", optimizer_kwargs={}, n_epochs=200, logger=None):
        optimizer = Optimizer(optimizer, self.parameters(), **optimizer_kwargs)

        for epoch in range(n_epochs):
            losses = []

            for step, (X, label) in tqdm(enumerate(dataloader), desc=f"Epoch {epoch}", total=len(dataloader)):
                optimizer.zero_grad()
                loss = self.compute_loss(X, label)
                losses.append(loss.item())
                loss.backward()
                optimizer.step()

            if logger:
                logger(epoch, losses)
