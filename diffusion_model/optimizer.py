import torch


def Optimizer(optimizer, parameters, **kwargs):
    if optimizer == "adam":
        return torch.optim.Adam(parameters, **kwargs)
    elif optimizer == "sgd":
        return torch.optim.SGD(parameters, **kwargs)
    else:
        raise ValueError('Unknown optimizer')
