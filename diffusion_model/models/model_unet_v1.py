import torch
from torch import nn

from ..scheduler import Scheduler
from ..forwarder import Forwarder
from ..loss import Loss
from ..constants import device


class Block(nn.Module):
    def __init__(self, in_ch, out_ch) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.act(x)
        return x


class Encoder(nn.Module):
    def __init__(self, chs):
        super().__init__()
        self.blocks = nn.ModuleList([Block(chs[i], chs[i+1])
                                     for i in range(len(chs)-1)])
        self.transforms = nn.ModuleList([nn.MaxPool2d(kernel_size=2, stride=2)
                                         for i in range(len(chs)-1)])

    def forward(self, x):
        residuals = []
        for block, transform in zip(self.blocks, self.transforms):
            x = block(x)
            residuals.append(x)
            x = transform(x)
        return residuals[-1], residuals[:-1]


class Decoder(nn.Module):
    def __init__(self, chs):
        super().__init__()
        self.blocks = nn.ModuleList([Block(chs[i], chs[i+1])
                                     for i in range(len(chs)-1)])
        self.transforms = nn.ModuleList([nn.ConvTranspose2d(chs[i], chs[i+1], kernel_size=2, stride=2)
                                         for i in range(len(chs)-1)])

    def forward(self, x, residuals):
        for block, transform in zip(self.blocks, self.transforms):
            x = transform(x)
            x = torch.cat([x, residuals.pop()], dim=1)
            x = block(x)
        return x


class UNet(nn.Module):
    def __init__(self, down_chs, up_chs):
        super().__init__()
        self.encoder = Encoder(down_chs)
        self.decoder = Decoder(up_chs[:-1])
        self.head = nn.Conv2d(
            up_chs[-2], up_chs[-1], kernel_size=3, padding="same")

    def forward(self, x):
        x, residuals = self.encoder(x)
        x = self.decoder(x, residuals)
        x = self.head(x)
        return x


class Model_UNet_V1(nn.Module):
    def __init__(self, scheduler: Scheduler, forwarder: Forwarder, chs=(32, 64, 128), *_, **__) -> None:
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

        label = nn.functional.one_hot(label, 10).float().to(device)
        label = self.label_embedding(label)
        label = label[:, None]

        input = torch.cat([X, label, t], dim=1)
        out = self.u_net(input)
        return out

    def _one_step(self, loss_fn: Loss, X, label):
        batch_size = X.shape[0]
        t = torch.randint(1, self.sch.n_steps+1, (batch_size, 1))

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
