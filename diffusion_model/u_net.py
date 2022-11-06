# inspired from https://amaarora.github.io/2020/09/13/unet.html, https://colab.research.google.com/drive/1sjy9odlSSy0RBVgMTgP7s99NXsqglsUL?usp=sharing#scrollTo=LQnlc27k7Aiw and https://huggingface.co/blog/annotated-diffusion

import torch
from torch import nn


class Block(nn.Module):
    def __init__(self, in_ch, out_ch) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.act(x)
        return x


class Encoder(nn.Module):
    def __init__(self, chs):
        super().__init__()
        self.blocks = nn.ModuleList([Block(chs[i], chs[i+1])
                                     for i in range(len(chs)-1)])
        self.transforms = nn.ModuleList([nn.MaxPool2d(kernel_size=2, stride=2)
                                         for i in range(len(chs)-1)])

    def forward(self, x):
        residuals = []
        for block, transform in zip(self.blocks, self.transforms):
            x = block(x)
            residuals.append(x)
            x = transform(x)
        return residuals[-1], residuals[:-1]


class Decoder(nn.Module):
    def __init__(self, chs):
        super().__init__()
        self.blocks = nn.ModuleList([Block(chs[i], chs[i+1])
                                     for i in range(len(chs)-1)])
        self.transforms = nn.ModuleList([nn.ConvTranspose2d(chs[i], chs[i+1], kernel_size=2, stride=2)
                                         for i in range(len(chs)-1)])

    def forward(self, x, residuals):
        for block, transform in zip(self.blocks, self.transforms):
            x = transform(x)
            x = torch.cat([x, residuals.pop()], dim=1)
            x = block(x)
        return x


class UNet(nn.Module):
    def __init__(self, down_chs, up_chs):
        super().__init__()
        self.encoder = Encoder(down_chs)
        self.decoder = Decoder(up_chs[:-1])
        self.head = nn.Conv2d(up_chs[-2], up_chs[-1], kernel_size=1)

    def forward(self, x):
        x, residuals = self.encoder(x)
        x = self.decoder(x, residuals)
        x = self.head(x)
        return x
