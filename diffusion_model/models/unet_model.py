# inspired from https://amaarora.github.io/2020/09/13/unet.html, https://colab.research.google.com/drive/1sjy9odlSSy0RBVgMTgP7s99NXsqglsUL?usp=sharing#scrollTo=LQnlc27k7Aiw and https://huggingface.co/blog/annotated-diffusion

import torch
from torch import nn
from einops import rearrange, repeat

from .base_model import BaseNoiseModel
from ..constants import device
from ..embeddings import LinearEmbedding


class Block(nn.Module):
    def __init__(self, in_ch, out_ch, up=False) -> None:
        super().__init__()
        # self.transform = nn.Conv2d(
        #     out_ch, out_ch, kernel_size=4, stride=2, padding=1)

        self.up = up
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.act = nn.ReLU()

        if up:
            self.transform = nn.ConvTranspose2d(
                out_ch, out_ch, kernel_size=2, stride=2)
        else:
            self.transform = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.bn1 = nn.ReLU()
        # self.bn2 = nn.ReLU()
        # self.act = nn.LeakyReLU(0.2)
        # self.scale_mlp = nn.Linear(emb_dim, out_ch)
        # self.shift_mlp = nn.Linear(emb_dim, out_ch)

    def forward(self, x, scale_shift=None):
        # print("block[1]", x.shape)
        print("conv", self.conv1)
        x = self.conv1(x)
        print("act", self.act)
        x = self.act(x)
        # print("block[2]", x.shape)
        # x = self.bn1(x)

        # if scale_shift:
        #     scale, shift = scale_shift
        #     scale = rearrange(self.scale_mlp(scale), "n c -> n c 1 1")
        #     shift = rearrange(self.shift_mlp(shift), "n c -> n c 1 1")
        #     x = scale * x + shift

        print("conv", self.conv2)
        x = self.conv2(x)
        print("act", self.act)
        x = self.act(x)
        # print("block[3]", x.shape)
        # x = self.bn2(x)

        print("transform", self.transform)
        x_next = self.transform(x)
        # print("block[4]", x.shape)

        return x, x_next


# class Encoder(nn.Module):
#     def __init__(self, chs, input_dim, emb_dim) -> None:
#         super().__init__()
#         self.blocks = nn.ModuleList(
#             [Block(chs[i], chs[i+1], input_dim//(2**i), emb_dim) for i in range(len(chs) - 1)])
#         self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

#     def forward(self, x, scale_shift=None):
#         features = []
        # print("encode", x.shape)
#         for block in self.blocks:
#             x = block(x, scale_shift)
        # print("encode", x.shape)
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
        # print("decode", x.shape)
#         for block, up_conv, enc_feature in zip(self.blocks, self.up_convs, enc_features):
        # print("decode[1]", x.shape, up_conv, enc_feature.shape)
#             x = up_conv(x)
        # print("decode[2]", x.shape)
#             x = torch.cat([enc_feature, x], dim=1)
        # print("decode[3]", x.shape)
#             x = block(x, scale_shift)
        # print("decode[4]", x.shape)
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
    def __init__(self, forward_module, scheduler, chs=(32, 64, 128)) -> None:
        super().__init__(forward_module, scheduler)
        print("V15")
        self.nb_steps = scheduler.nb_steps
        # self.nb_labels = 10

        # emb_dim = 16
        down_chs = [1, *chs]
        up_chs = chs[::-1]

        # self.time_mlp = nn.Sequential(
        #     nn.Linear(1, 28*28),
        #     nn.ReLU(),
        #     nn.Linear(28*28, 28*28),
        #     nn.ReLU(),
        # ).to(device)
        # self.label_mlp = nn.Sequential(
        #     nn.Linear(self.nb_labels, 28*28),
        #     nn.ReLU(),
        #     nn.Linear(28*28, 28*28),
        #     nn.ReLU(),
        # ).to(device)

        # self.conv0 = nn.Conv2d(2, chs[0], kernel_size=3, padding=1).to(device)

        self.downs = nn.ModuleList([
            Block(down_chs[i], down_chs[i+1], up=False) for i in range(len(down_chs) - 1)
        ]).to(device)
        self.middle = Block(up_chs[0], up_chs[1], up=True).to(device)
        self.ups = nn.ModuleList([
            Block(2*up_chs[i], up_chs[i+1], up=True) for i in range(1, len(up_chs) - 1)
        ]).to(device)

        self.output = nn.Conv2d(up_chs[-1], 1, kernel_size=1).to(device)

    def forward(self, x, time, label):
        x = rearrange(x, "b h w -> b 1 h w")
        time = rearrange(time, "b -> b 1")
        # label = rearrange(label, "b -> b 1")

        time = rescale(time, (0, self.nb_steps-1), (0, 1))
        time = repeat(time, "b 1 -> b 1 h w", h=28, w=28)
        # time = self.time_mlp(time)

        # label = one_hot_encode(label, self.nb_labels)
        # label = self.label_mlp(label)
        # label = rearrange(label, "b (h w) -> b 1 h w", h=28)

        # x = torch.cat([x], dim=1)

        # x = self.conv0(x)

        residual_inputs = []
        for down in self.downs:
            print("down[1]", x.shape)
            x, x_next = down(x)
            residual_inputs.append(x)
            print("down[2]", x_next.shape)
            x = x_next

        x = residual_inputs.pop()
        print("middle", x.shape)
        _, x = self.middle(x)
        # print(len(residual_inputs))

        for up, res_input in zip(self.ups[::-1], residual_inputs[::-1]):
            x = torch.cat([x, res_input], dim=1)
            print("up[1]", x.shape, res_input.shape)
            _, x = up(x)
            print("up[2]", x.shape, res_input.shape)

        x = self.output(x)
        x = rearrange(x, "b 1 h w -> b h w")
        return x


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
# for ftr in ftrs: print(ftr.shape)
# print(x.shape)

# # decoder = Decoder((1,64,128,256,512,1024)[::-1])
# # x = torch.randn(1, 1024, 28, 28)
# # decoder(x, ftrs[:-1][::-1]).shape


# # x_prime = torch.randn(10, 2, 1, 2)
# # model = UNet(in_chs=(2, 64), out_chs=(64, 1)).to(device)
# # model(x_prime).shape
