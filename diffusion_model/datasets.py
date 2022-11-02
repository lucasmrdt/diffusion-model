import torch
from torchvision import datasets
from sklearn.datasets import make_moons
import numpy as np

from .constants import device


def _normalize(X):
    X = X - X.min(axis=0) / (X.max(axis=0) -
                             X.min(axis=0))  # normalize to [0, 1]
    X = X * 2 - 1  # normalize to [-1, 1]

    return X


def get_moons_dataset(n_samples, noise=0.05):
    X, _ = make_moons(n_samples=n_samples, noise=noise)
    X = _normalize(X)
    X = torch.from_numpy(X).float().to(device)
    return X


def get_mnist_dataset(n_samples):
    X = datasets.MNIST(root="../datasets", download=True, train=True)
    X = X.data.float().to(device)
    X = X[torch.randperm(X.shape[0])][:n_samples]
    X = 2 * X / 255 - 1  # normalize to [-1, 1]
    return X
