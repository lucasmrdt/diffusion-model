# inspired from https://github.com/acids-ircam/diffusion_models

from torch import nn
import torch.nn.functional as F

from .base_model import BaseNoiseModel


class ConditionalLinear(nn.Module):
    def __init__(self, num_in, num_out, nb_steps):
        super().__init__()
        self.num_out = num_out
        self.lin = nn.Linear(num_in, num_out)
        self.embed = nn.Embedding(nb_steps, num_out)
        self.embed.weight.data.uniform_()

    def forward(self, x, t):
        out = self.lin(x)
        gamma = self.embed(t)
        out = gamma.view(-1, self.num_out) * out
        return out


class ConditionalNoiseModel(BaseNoiseModel):
    def __init__(self, forward_module, scheduler):
        super().__init__(forward_module, scheduler)
        input_dim = scheduler.input_dim
        nb_steps = scheduler.nb_steps
        self.lin1 = ConditionalLinear(input_dim, 128, nb_steps)
        self.lin2 = ConditionalLinear(128, 128, nb_steps)
        self.lin3 = ConditionalLinear(128, 128, nb_steps)
        self.lin4 = nn.Linear(128, input_dim)

    def forward(self, x, t):
        x = F.softplus(self.lin1(x, t))
        x = F.softplus(self.lin2(x, t))
        x = F.softplus(self.lin3(x, t))
        return self.lin4(x)
