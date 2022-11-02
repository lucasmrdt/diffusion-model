import torch
from torch import nn
from einops import repeat
from tqdm import tqdm

from .constants import device


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
    def __init__(self, dim, min_val, max_val):
        super().__init__()
        self.dim = dim
        self.min_val = min_val
        self.max_val = max_val

    def forward(self, x):
        w, h = self.dim
        x = 2 * (x - self.min) / (self.max_val - self.min_val) - 1  # [-1, 1]
        x = repeat(x, 'n -> n w h', w=w, h=h)
        return x


class MLPEmbedding(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss()
        self.embedding = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, input_dim)
        ).to(device)
        self.nn = nn.Sequential(self.embedding, nn.Softmax()).to(device)

    def forward(self, x):
        return self.nn(x)

    def fit(self, dataloader, nb_epochs=200):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

        for epoch in range(nb_epochs):
            losses = []

            for i, (X, y) in tqdm(enumerate(dataloader)):
                optimizer.zero_grad()
                loss = self.loss_fn(self.forward(X), y)
                losses.append(loss.item())
                loss.backward()
                optimizer.step()

            if i % (nb_epochs // 10) == 0:
                print(f"[Epoch {epoch}]: loss={sum(losses)/len(losses)}")
