from .model_unet_v1 import Model_UNet_V1
from .model_dense import model_dense

class ModelGetter:
    valid_choices = [
        "unet_v1",
    ]
    default = "unet_v1"

    @staticmethod
    def get_model(model):
        if model == "unet_v1":
            return Model_UNet_V1
        elif model == "dense":
            return model_dense
        else:
            raise ValueError(f"Unknown model: {model}")
