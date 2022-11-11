from .model_unet_v1 import Model_UNet_V1


class ModelGetter:
    valid_choices = [
        "unet_v1",
    ]
    default = "unet_v1"

    @staticmethod
    def get_model(model):
        if model == "unet_v1":
            return Model_UNet_V1
        else:
            raise ValueError(f"Unknown model: {self.model}")
