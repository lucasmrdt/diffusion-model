import torch
from torch import nn

from ..scheduler import Scheduler
from ..forwarder import Forwarder
from ..loss import Loss
from ..constants import device

class linear_relu(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.input = input_dim
        self.output = output_dim

        self.linear = nn.Linear(self.input, self.output)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.linear(x)
        out = self.act(x)
        return out


class model_dense(nn.Module):

    def __init__(self, scheduler: Scheduler, forwarder: Forwarder, depth: int = 4, width: int = 512, dropout: float = 0.3, *_, **__):
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
            nn.Flatten(),
            nn.Linear((32*32)*3,width),
            nn.ReLU(),
        )
        self.hidden = nn.ModuleList([linear_relu(width, width) for _ in range(depth)])
        self.output = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(width, 32*32),
            nn.Unflatten(1, (32, 32)),
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
            x = self.layer(x)
        out = self.output(x)

        return out

    def _one_step(self, loss_fn: Loss, x, label):
        batch_size = x.shape[0]
        t = torch.randint(1, self.sch.n_steps+1, (batch_size, 1))

        x, label, t = X.to(device), label.to(device), t.to(device)

        x_noisy, noise = self.fwd.forward(X, t)
        model_pred = self.forward(x_noisy, t, label)

        return loss_fn(t, x_noisy, noise, model_pred)

    def one_step_eval(self, loss_fn: Loss, x, label):
        loss = self._one_step(loss_fn, x, label)
        return loss.item()

    def one_step_training(self, optimizer: torch.optim.Optimizer, loss_fn: Loss, x, label):
        optimizer.zero_grad()

        loss = self._one_step(loss_fn, x, label)
        loss.backward()
        optimizer.step()

        return loss.item()
