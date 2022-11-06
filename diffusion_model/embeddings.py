import torch
from torch import nn
from einops import repeat
from tqdm import tqdm

from .constants import device

def rescale(x, in_bound, out_bound):
    x = (x - in_bound[0])/(in_bound[1] - out_bound[0])
    x = x * (out_bound[1] - out_bound[0]) + out_bound[0]
    return x


class ScalingSigmoid(nn.Module):
    def __init__(self, min_val, max_val):
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val

    def forward(self, x):
        return rescale(torch.sigmoid(x), (0, 1), (self.min_val, self.max_val))

class SinusoidalEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = torch.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(
            half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class LinearEmbedding(nn.Module):
    def __init__(self, dim, min_val=0, max_val=1):
        super().__init__()
        self.dim = dim
        self.min_val = min_val
        self.max_val = max_val

    def forward(self, x):
        x = 2 * (x-self.min_val) / (self.max_val-self.min_val) - 1  # [-1, 1]
        # x = (x-self.min_val) / (self.max_val-self.min_val)  # [0, 1]
        x = repeat(x, "n -> n d", d=self.dim)
        return x


class MLPEmbedding(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.loss_fn = nn.functional.binary_cross_entropy
        self.embedding = nn.Sequential(
            nn.Linear(input_dim, output_dim),
        ).to(device)
        self.nn = nn.Sequential(
            self.embedding,
            nn.ReLU(),
            nn.Linear(output_dim, input_dim),
            nn.Softmax(dim=-1)
        ).to(device)

    def forward(self, x, training=False):
        if training:
            return self.nn(x)
        else:
            # return self.embedding(x)
            return self.embedding(x)

    def fit(self, dataloader, nb_epochs=200):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        last_loss = None

        for _ in tqdm(range(nb_epochs)):
            losses = []
            for X in dataloader:
                optimizer.zero_grad()
                X_pred = self.forward(X, training=True)
                loss = self.loss_fn(X_pred, X)
                losses.append(loss.item())
                loss.backward()
                optimizer.step()
            last_loss = sum(losses) / len(losses)

        return last_loss
