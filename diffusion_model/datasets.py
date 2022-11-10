import torch
import torch.nn as nn
from functools import partial
from torchvision import datasets, transforms

from .constants import device, DATASETS_DIR


class Normalize(nn.Module):
    def __init__(self, normalize_range):
        super().__init__()
        self.normalize_range = normalize_range

    def forward(self, x):
        o_min, o_max = self.normalize_range
        x = x * (o_max - o_min) + o_min  # scale to [o_min, o_max]
        return x


def get_mnist_dataset(normalize_range=(-1, 1), batch_size=256):
    transform = transforms.Compose([
        transforms.Pad(padding=2),
        transforms.ToTensor(),
        Normalize(normalize_range),
    ])

    train = datasets.MNIST(root=DATASETS_DIR, download=True,
                           train=True, transform=transform)
    test = datasets.MNIST(root=DATASETS_DIR, download=True,
                          train=False, transform=transform)

    train_loader = torch.utils.data.DataLoader(
        train, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        test, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
