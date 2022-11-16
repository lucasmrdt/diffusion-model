import torch
from torch import nn

from ..scheduler import Scheduler
from ..forwarder import Forwarder
from ..loss import Loss
from ..constants import device


class Conv2D_relu(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input = input_dim
        self.output = output_dim
        self.conv = nn.Conv2d(self.input, self.output, 3, padding="same")
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        out = self.act(x)
        return out


class model_Conv2D(nn.Module):

    def __init__(self, scheduler: Scheduler, forwarder: Forwarder, depth: int = 4, width: int = 32, dropout: float = 0.3, *_, **__):
        super().__init__()

        self.sch = scheduler
        self.fwd = forwarder

        self.time_embedding = nn.Sequential(
            nn.Linear(1, 32 * 32),
            nn.ReLU(),
            nn.Unflatten(1, (32, 32)),
        )

        self.label_embedding = nn.Sequential(
            nn.Linear(10, 32 * 32),
            nn.ReLU(),
            nn.Unflatten(1, (32, 32)),
        )

        self.input = nn.Sequential(
            nn.Conv2d(3, depth, 3, padding="same"),
            nn.ReLU(),
        )
        self.hidden = nn.ModuleList([Conv2D_relu(width, width) for _ in range(depth)])
        self.output = nn.Sequential(
            nn.Dropout(dropout),
            nn.Conv2d(width, 1, 3, padding="same"),
        )

    def forward(self, x, t, label):
        t = 2 * t / self.sch.n_steps - 1  # [-1, 1]
        t = self.time_embedding(t)
        t = t[:, None]

        label = nn.functional.one_hot(label, 10).float().to(device)
        label = self.label_embedding(label)
        label = label[:, None]

        input = torch.cat([x, label, t], dim=1)

        x = self.input(input)
        for layer in self.hidden:
            x = layer(x)
        out = self.output(x)

        return out
