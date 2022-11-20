from .model_unet_v1 import Model_UNet_V1
from .model_dense import model_dense
from .model_Conv2D import model_Conv2D


class ModelGetter:
    valid_choices = [
        "unet_v1",
        "dense",
        "conv"
    ]
    default = "unet_v1"

    @staticmethod
    def get_model(model):
        if model == "unet_v1":
            return Model_UNet_V1
        elif model == "dense":
            return model_dense
        elif model == "conv":
            return model_Conv2D
        else:
            raise ValueError(f"Unknown model: {model}")
