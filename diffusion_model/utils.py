import torch
from .constants import device, SEED
import random
import numpy as np


def check_k_last_increasing(l, k):
    """Check if k last elements of l are increasing.

    Args:
        l (List[int]): The list to check.
        k (int): The number of elements to check.

    Returns:
        bool: Whether the k last elements of l are increasing.
    """
    N = len(l)
    return N >= k and all(x1 < x2 for x1, x2 in zip(l[N-k:], l[N+1-k:]))


def random_seed(seed=None):
    seed = SEED if seed is None else seed
    random.seed(seed)  # To ensure the same data split between experiments.
    np.random.seed(seed)
    torch.random.manual_seed(seed)
