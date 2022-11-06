import torch
from torchvision import datasets
from .constants import device


def get_mnist_dataset(n_samples=None):
    X = datasets.MNIST(root="../datasets", download=True, train=True)
    X, y = X.data.float().to(device), X.targets.to(device)

    if n_samples is not None:
        perm = torch.randperm(X.shape[0])
        X = X[perm][:n_samples]
        y = y[perm][:n_samples]

    X = 2 * X / 255 - 1  # [-1, 1]
    return X, y
