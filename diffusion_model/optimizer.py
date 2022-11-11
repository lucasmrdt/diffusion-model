import torch


class OptimizerGetter:
    valid_choices = ["adam", "sgd"]
    default = "adam"

    @staticmethod
    def get_optimizer(optimizer):
        if optimizer == "adam":
            return torch.optim.Adam
        elif optimizer == "sgd":
            return torch.optim.SGD
        else:
            raise ValueError('Unknown optimizer')
