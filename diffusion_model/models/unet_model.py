# inspired from https://amaarora.github.io/2020/09/13/unet.html, https://colab.research.google.com/drive/1sjy9odlSSy0RBVgMTgP7s99NXsqglsUL?usp=sharing#scrollTo=LQnlc27k7Aiw and https://huggingface.co/blog/annotated-diffusion

import torch
from torch import nn
from einops import rearrange, repeat

from .base_model import BaseNoiseModel
from ..constants import device
from ..embeddings import LinearEmbedding


class Block(nn.Module):
    def __init__(self, in_ch, out_ch, emb_dim, up=False) -> None:
        super().__init__()
        # print(in_ch, out_ch, input_dim)
        if up:
            self.conv1 = nn.Conv2d(2*in_ch, out_ch, kernel_size=3, padding=1)
            self.transform = nn.ConvTranspose2d(
                out_ch, out_ch, kernel_size=2, stride=2)
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
            self.transform = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.bn2 = nn.BatchNorm2d(out_ch)
        # self.act = nn.LeakyReLU(0.2)
        self.scale_mlp = nn.Linear(emb_dim, out_ch)
        self.shift_mlp = nn.Linear(emb_dim, out_ch)
        self.act = nn.ReLU()

    def forward(self, x, scale_shift=None):
        x = self.conv1(x)
        x = self.act(x)
        x = self.bn1(x)

        if scale_shift:
            scale, shift = scale_shift
            scale = rearrange(self.scale_mlp(scale), "n c -> n c 1 1")
            shift = rearrange(self.shift_mlp(shift), "n c -> n c 1 1")
            x = scale * x + shift

        x = self.conv2(x)
        x = self.act(x)
        x = self.bn2(x)

        return x


# class Encoder(nn.Module):
#     def __init__(self, chs, input_dim, emb_dim) -> None:
#         super().__init__()
#         self.blocks = nn.ModuleList(
#             [Block(chs[i], chs[i+1], input_dim//(2**i), emb_dim) for i in range(len(chs) - 1)])
#         self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

#     def forward(self, x, scale_shift=None):
#         features = []
#         # print("encode", x.shape)
#         for block in self.blocks:
#             x = block(x, scale_shift)
#             # print("encode", x.shape)
#             features.append(x)
#             x = self.pool(x)
#         return features[-1], features


# class Decoder(nn.Module):
#     def __init__(self, chs, input_dim, emb_dim) -> None:
#         super().__init__()
#         self.blocks = nn.ModuleList(
#             [Block(chs[i], chs[i+1], int(input_dim//(2**(len(chs)-3-i))), emb_dim) for i in range(len(chs) - 1)])
#         self.up_convs = nn.ModuleList([nn.ConvTranspose2d(
#             chs[i], chs[i+1], kernel_size=2, stride=2) for i in range(len(chs) - 1)])

#     def forward(self, x, enc_features, scale_shift=None):
#         # print("decode", x.shape)
#         for block, up_conv, enc_feature in zip(self.blocks, self.up_convs, enc_features):
#             # print("decode[1]", x.shape, up_conv, enc_feature.shape)
#             x = up_conv(x)
#             # print("decode[2]", x.shape)
#             x = torch.cat([enc_feature, x], dim=1)
#             # print("decode[3]", x.shape)
#             x = block(x, scale_shift)
#             # print("decode[4]", x.shape)
#         return x

def rescale(x, in_bound, out_bound):
    x = (x - in_bound[0])/(in_bound[1] - out_bound[0])
    x = x * (out_bound[1] - out_bound[0]) + out_bound[0]
    return x


def one_hot_encode(x, size):
    x = x.long()
    x = torch.zeros(x.size(0), size).to(device).scatter_(1, x, 1)
    return x


class UNetNoiseModel(BaseNoiseModel):
    def __init__(self, forward_module, scheduler, chs=(64, 128, 256)) -> None:
        super().__init__(forward_module, scheduler)
        self.nb_steps = scheduler.nb_steps
        self.nb_labels = 10

        emb_dim = 32
        down_chs = chs
        up_chs = chs[::-1]

        self.time_mlp = nn.Sequential(
            nn.Linear(1, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
        )
        self.label_mlp = nn.Sequential(
            nn.Linear(self.nb_labels, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
        )

        self.conv0 = nn.Conv2d(1, chs[0], kernel_size=3, padding=1)

        self.downs = nn.ModuleList([
            Block(down_chs[i], down_chs[i+1], emb_dim, up=False) for i in range(len(down_chs) - 1)
        ])
        self.ups = nn.ModuleList([
            Block(up_chs[i], up_chs[i+1], emb_dim, up=True) for i in range(len(up_chs) - 1)
        ])

        self.output = nn.Conv2d(up_chs[-1], 1, kernel_size=1)

    def forward(self, x, time, label):
        time = self.time_mlp(rescale(time, (0, self.nb_steps-1), (-1, 1)))
        label = self.label_mlp(one_hot_encode(label, self.nb_labels))

        x = self.conv0(x)

        residual_inputs = []
        for down in self.downs:
            x = down(x, (time, label))
            residual_inputs.append(x)
        for up, res_input in zip(self.ups, residual_inputs[::-1]):
            x = torch.cat([x, res_input], dim=1)
            x = up(x, (time, label))

        return self.output(x)


# class UNetNoiseModel(BaseNoiseModel):
#     def __init__(self, forward_module, scheduler, time_embedding, label_embedding):
#         super().__init__(forward_module, scheduler)
#         self.nb_steps = scheduler.nb_steps
#         self.input_dim = scheduler.input_dim
#         self.time_embedding = time_embedding
#         self.label_embedding = label_embedding

#         assert self.input_dim[0] == self.input_dim[1], "input_dim must be square"
#         dim = self.input_dim[0]

#         # common_chs = (64, 128, 256, 512, 1024)
#         common_chs = (64, 128, 256)
#         self.unet = UNet(in_chs=(3, *common_chs),
#                          out_chs=(*common_chs[::-1], 1),
#                          input_dim=dim).to(device)

#     def forward(self, x, time_step, label):
#         time_step = self.time_embedding(time_step)
#         label = self.label_embedding(label)

#         h, w = self.input_dim
#         x = rearrange(x, "n h w -> n 1 h w")
#         time_step = rearrange(time_step, "n (h w) -> n 1 h w", h=h)
#         label = rearrange(label, "n (h w) -> n 1 h w", h=h)
#         x = torch.cat([x, time_step, label], dim=1)

#         x = self.unet(x, scale_shift=(time_step, label))
#         x = rearrange(x, "d () h w -> d h w")
#         return x

# # encoder = Encoder((1,64,128,256,512,1024))
# # # input image
# # x, ftrs = encoder(x)
# # for ftr in ftrs: print(ftr.shape)
# # print(x.shape)

# # decoder = Decoder((1,64,128,256,512,1024)[::-1])
# # x = torch.randn(1, 1024, 28, 28)
# # decoder(x, ftrs[:-1][::-1]).shape


# # x_prime = torch.randn(10, 2, 1, 2)
# # model = UNet(in_chs=(2, 64), out_chs=(64, 1)).to(device)
# # model(x_prime).shape
