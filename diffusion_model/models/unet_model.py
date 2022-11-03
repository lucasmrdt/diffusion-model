# inspired from https://amaarora.github.io/2020/09/13/unet.html, https://colab.research.google.com/drive/1sjy9odlSSy0RBVgMTgP7s99NXsqglsUL?usp=sharing#scrollTo=LQnlc27k7Aiw and https://huggingface.co/blog/annotated-diffusion

import torch
from torch import nn
from einops import rearrange, repeat

from .base_model import BaseNoiseModel
from ..constants import device


class Block(nn.Module):
    def __init__(self, in_ch, out_ch, input_dim, feature_dim) -> None:
        super().__init__()
        # print(in_ch, out_ch, input_dim)
        self.input_dim = input_dim
        self.feature_dim = feature_dim

        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.relu = nn.LeakyReLU(0.2)
        self.scale_mlp = nn.Linear(feature_dim**2, input_dim**2)
        self.shift_mlp = nn.Linear(feature_dim**2, input_dim**2)

        if out_ch % 2 == 0:
            self.norm = nn.LeakyReLU(0.2)
            # self.norm = nn.GroupNorm(8, out_ch)
        else:
            self.norm = nn.Identity()

    def forward(self, x, scale_shift=None):
        x = self.conv1(x)
        x = self.norm(x)

        if scale_shift:
            c, h, w = x.shape[1:]
            scale, shift = scale_shift
            # print("before", x.shape, scale.shape, shift.shape,
            #       self.input_dim, self.feature_dim, (c, h, w))

            scale = self.scale_mlp(rearrange(scale, "n 1 h w -> n (h w)"))
            # print("before[1]", scale.shape)
            scale = rearrange(scale, "n (h w) -> n 1 h w", h=h, w=w)
            scale = repeat(scale, "n 1 h w -> n c h w", c=c)

            shift = self.shift_mlp(rearrange(shift, "n 1 h w -> n (h w)"))
            shift = rearrange(shift, "n (h w) -> n 1 h w", h=h)
            shift = repeat(shift, "n 1 h w -> n c h w", c=c)
            # print("after", x.shape, scale.shape, shift.shape,
            #       self.input_dim, self.feature_dim, (c, h, w))

            x = scale * x + shift

        # x = self.relu(x)
        # x = self.bn(x)
        x = self.conv2(x)
        x = self.relu(x)
        # x = self.bn(x)
        return x


class Encoder(nn.Module):
    def __init__(self, chs, input_dim, feature_dim) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [Block(chs[i], chs[i+1], input_dim//(2**i), feature_dim) for i in range(len(chs) - 1)])
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x, scale_shift=None):
        features = []
        # print("encode", x.shape)
        for block in self.blocks:
            x = block(x, scale_shift)
            # print("encode", x.shape)
            features.append(x)
            x = self.pool(x)
        return features[-1], features


class Decoder(nn.Module):
    def __init__(self, chs, input_dim, feature_dim) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [Block(chs[i], chs[i+1], int(input_dim//(2**(len(chs)-3-i))), feature_dim) for i in range(len(chs) - 1)])
        self.up_convs = nn.ModuleList([nn.ConvTranspose2d(
            chs[i], chs[i+1], kernel_size=2, stride=2) for i in range(len(chs) - 1)])

    def forward(self, x, enc_features, scale_shift=None):
        # print("decode", x.shape)
        for block, up_conv, enc_feature in zip(self.blocks, self.up_convs, enc_features):
            # print("decode[1]", x.shape, up_conv, enc_feature.shape)
            x = up_conv(x)
            # print("decode[2]", x.shape)
            x = torch.cat([enc_feature, x], dim=1)
            # print("decode[3]", x.shape)
            x = block(x, scale_shift)
            # print("decode[4]", x.shape)
        return x


class UNet(nn.Module):
    def __init__(self, in_chs, out_chs, input_dim) -> None:
        super().__init__()
        self.encoder = Encoder(in_chs, input_dim, feature_dim=input_dim)
        self.decoder = Decoder(out_chs, input_dim, feature_dim=input_dim)
        self.head = nn.Conv2d(out_chs[-2], out_chs[-1], kernel_size=1)

    def forward(self, x, scale_shift=None):
        x, enc_features = self.encoder(x, scale_shift)
        x = self.decoder(x, enc_features[:-1][::-1], scale_shift)
        x = self.head(x)
        return x


class UNetNoiseModel(BaseNoiseModel):
    def __init__(self, forward_module, scheduler, time_embedding, label_embedding):
        super().__init__(forward_module, scheduler)
        self.nb_steps = scheduler.nb_steps
        self.input_dim = scheduler.input_dim
        self.time_embedding = time_embedding
        self.label_embedding = label_embedding

        assert self.input_dim[0] == self.input_dim[1], "input_dim must be square"
        dim = self.input_dim[0]

        # common_chs = (64, 128, 256, 512, 1024)
        common_chs = (64, 128, 256)
        self.unet = UNet(in_chs=(3, *common_chs),
                         out_chs=(*common_chs[::-1], 1),
                         input_dim=dim).to(device)

    def forward(self, x, time_step, label):
        time_step = self.time_embedding(time_step)
        label = self.label_embedding(label)

        h, w = self.input_dim
        x = rearrange(x, "n h w -> n 1 h w")
        time_step = rearrange(time_step, "n (h w) -> n 1 h w", h=h)
        label = rearrange(label, "n (h w) -> n 1 h w", h=h)
        x = torch.cat([x, time_step, label], dim=1)

        x = self.unet(x, scale_shift=(time_step, label))
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
