# inspired from https://amaarora.github.io/2020/09/13/unet.html, https://colab.research.google.com/drive/1sjy9odlSSy0RBVgMTgP7s99NXsqglsUL?usp=sharing#scrollTo=LQnlc27k7Aiw and https://huggingface.co/blog/annotated-diffusion

import torch
from torch import nn
from einops import rearrange

from .base_model import BaseNoiseModel
from ..embeddings import LinearEmbedding, SinusoidalEmbeddings
from ..constants import device


class Block(nn.Module):
    def __init__(self, in_ch, out_ch) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.act = nn.SiLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.act(x)
        return x


class Encoder(nn.Module):
    def __init__(self, chs) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [Block(chs[i], chs[i+1]) for i in range(len(chs) - 1)])
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        features = []
        # print("encode", x.shape)
        for block in self.blocks:
            x = block(x)
            # print("encode", x.shape)
            features.append(x)
            x = self.pool(x)
        return features[-1], features


class Decoder(nn.Module):
    def __init__(self, chs) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [Block(chs[i], chs[i+1]) for i in range(len(chs) - 1)])
        self.up_convs = nn.ModuleList([nn.ConvTranspose2d(
            chs[i], chs[i+1], kernel_size=2, stride=2) for i in range(len(chs) - 1)])

    def forward(self, x, enc_features):
        # print("decode", x.shape)
        for block, up_conv, enc_feature in zip(self.blocks, self.up_convs, enc_features):
            # print("decode[1]", x.shape, up_conv, enc_feature.shape)
            x = up_conv(x)
            # print("decode[2]", x.shape)
            x = torch.cat([enc_feature, x], dim=1)
            # print("decode[3]", x.shape)
            x = block(x)
            # print("decode[4]", x.shape)
        return x


class UNet(nn.Module):
    def __init__(self, in_chs, out_chs) -> None:
        super().__init__()
        self.encoder = Encoder(in_chs)
        self.decoder = Decoder(out_chs)
        self.head = nn.Conv2d(out_chs[-2], out_chs[-1], kernel_size=1)

    def forward(self, x):
        x, enc_features = self.encoder(x)
        x = self.decoder(x, enc_features[:-1][::-1])
        x = self.head(x)
        return x


class UNetNoiseModel(BaseNoiseModel):
    def __init__(self, forward_module, scheduler, time_embedding, label_embedding):
        super().__init__(forward_module, scheduler)
        self.nb_steps = scheduler.nb_steps
        self.input_dim = scheduler.input_dim
        self.time_embedding = time_embedding
        self.label_embedding = label_embedding

        # common_chs = (64, 128, 256, 512, 1024)
        common_chs = (32, 64, 128)
        self.unet = UNet(in_chs=(2, *common_chs),
                         out_chs=(*common_chs[::-1], 1)).to(device)

    def forward(self, x, time_step, label):
        time_step = self.embedding(time_step)
        label = self.embedding(time_step)

        x = rearrange(x, "n h w -> n () h w")
        time_step = rearrange(time_step, "n h w -> n () h w")
        label = rearrange(label, "n h w -> n () h w")
        x = torch.cat([x, time_step, label], dim=1)

        x = self.unet(x)
        x = rearrange(x, "d () h w -> d h w")
        return x

# encoder = Encoder((1,64,128,256,512,1024))
# # input image
# x, ftrs = encoder(x)
# for ftr in ftrs: print(ftr.shape)
# print(x.shape)

# decoder = Decoder((1,64,128,256,512,1024)[::-1])
# x = torch.randn(1, 1024, 28, 28)
# decoder(x, ftrs[:-1][::-1]).shape


# x_prime = torch.randn(10, 2, 1, 2)
# model = UNet(in_chs=(2, 64), out_chs=(64, 1)).to(device)
# model(x_prime).shape
