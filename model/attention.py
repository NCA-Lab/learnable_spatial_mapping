from turtle import forward
from typing import List
import torch
from torch import nn
import torch.nn.functional as F
from model.utils import *


class bandAttention3D(nn.Module):
    def __init__(self, input, reduction) -> None:
        super().__init__()
        # batch * band * frame * c1 * c2
        avgp = nn.AvgPool3d((input.shape[-3], input.shape[-2], input.shape[-1]))
        maxp = nn.MaxPool3d((input.shape[-3], input.shape[-2], input.shape[-1]))
        Favg = avgp(input).squeeze()
        Fmax = maxp(input).squeeze()
        self.add_module("avgp", avgp)
        self.add_module("maxp", maxp)
        fcnet = nn.Sequential(
            nn.Linear(input.shape[1], input.shape[1] // reduction),
            nn.Tanh(),
            nn.Dropout(),
            nn.Linear(input.shape[1] // reduction, input.shape[1]),
            nn.Tanh(),
            nn.Dropout(),
        )
        self.add_module("fcnet", fcnet)
        Favg = fcnet(Favg)
        Fmax = fcnet(Fmax)
        Fmask = torch.sigmoid(Favg + Fmax)
        input = input * Fmask.unsqueeze(dim=-1).unsqueeze(dim=-1).unsqueeze(dim=-1)

    def forward(self, input):
        M = torch.sigmoid(
            self.fcnet(self.avgp(input).squeeze())
            + self.fcnet(self.maxp(input).squeeze())
        )
        input = input * M.unsqueeze(dim=-1).unsqueeze(dim=-1).unsqueeze(dim=-1)
        return input, M


class bandAttention(nn.Module):
    def __init__(self, input, reduction, device='cpu') -> None:
        super().__init__()
        # batch * band * frame * channel
        avgp = nn.AvgPool2d((input.shape[-2], input.shape[-1])).to(device)
        maxp = nn.MaxPool2d((input.shape[-2], input.shape[-1])).to(device)
        Favg = avgp(input).squeeze()
        Fmax = maxp(input).squeeze()
        self.add_module("avgp", avgp)
        self.add_module("maxp", maxp)
        fcnet = nn.Sequential(
            nn.Linear(input.shape[1], input.shape[1] // reduction),
            nn.Tanh(),
            nn.Dropout(),
            nn.Linear(input.shape[1] // reduction, input.shape[1]),
            nn.Tanh(),
            nn.Dropout(),
        ).to(device)
        self.add_module("fcnet", fcnet)
        Favg = fcnet(Favg)
        Fmax = fcnet(Fmax)
        Fmask = torch.sigmoid(Favg + Fmax)
        input = input * Fmask.unsqueeze(dim=-1).unsqueeze(dim=-1)

    def forward(self, input):
        M = torch.sigmoid(
            self.fcnet(self.avgp(input).squeeze())
            + self.fcnet(self.maxp(input).squeeze())
        )
        input = input * M.unsqueeze(dim=-1).unsqueeze(dim=-1)
        return input


class channelAttention(nn.Module):
    def __init__(
        self, input, kernelSize=(1, 3, 3), imageSize=(8, 8,), device='cpu'
    ) -> None:
        super().__init__()
        # batch * band * frame * channel
        print("Initializing channel attention")
        print("Input size is")
        print(input.shape)
        # input = networkChMap(input)
        net = mapImage_mod(input, imageSize, device=device)
        input = net(input)
        self.add_module("map", net)
        # batch * band *frame * c1*c2
        input = input.permute((0, 2, 1, 3, 4))
        # batch * frame * band * c1 * c2
        a = nn.AvgPool3d((input.shape[2], 1, 1)).to(device)
        m = nn.MaxPool3d((input.shape[2], 1, 1)).to(device)
        Fa = a(input)
        Fm = m(input)
        # feature: batch * time * 1 * c1 * c2
        f = torch.concat((Fa, Fm), 2)
        # feature: batch * time * 2 * c1 * c2
        f = f.permute((0, 2, 1, 3, 4))
        # feature : batch * 2 * time * c1 * c2
        # f = f.permute((0, 3, 1, 2))
        convnet = nn.Sequential(
            nn.Conv3d(
                f.shape[1], 1, (1, kernelSize[1], kernelSize[2]), padding="same",
            ),
            nn.ELU(),
            nn.BatchNorm3d(1),
            nn.Dropout(),
            nn.AvgPool3d((f.shape[2], 1, 1)),
        ).to(device)
        # feature: batch * 1 * 1 * c1 * c2
        f = torch.sigmoid(convnet(f))
        input = input.permute((0, 2, 1, 3, 4)) * f
        # input : batch * band * frame * c1 * c2
        self.add_module("avgp", a)
        self.add_module("maxp", m)
        self.add_module('conv', convnet)
        print("Output size is")
        print(input.shape)
        # bf * band * c1 * c2

    def forward(self, input):
        # batch * band * frame * channel
        # input = networkChMap(input)
        input = self.map(input)
        # batch * band *frame * c1*c2
        input = input.permute((0, 2, 1, 3, 4))
        # batch * frame * band * c1 * c2
        Fa = self.avgp(input)
        Fm = self.maxp(input)
        # feature: batch * time * 1 * c1 * c2
        f = torch.concat((Fa, Fm), 2)
        # feature: batch * time * 2 * c1 * c2
        f = f.permute((0, 2, 1, 3, 4))
        # feature * batch * 2 * time * c1 * c2
        f = torch.sigmoid(self.conv(f))
        # feature: batch * 1 * time * c1 * c2
        input = input.permute((0, 2, 1, 3, 4)) * f
        # input : batch * band * frame * c1 * c2
        return input


# class channelAttention_fc(nn.Module):
#     def __init__(
#         self,
#         input,
#         kernelSize=(1, 3, 3),
#         imageSize=(8, 8,),
#         numFCNeurons=[0,],
#         device='cpu',
#     ) -> None:
#         super().__init__()
#         # batch * band * frame * channel
#         print("Initializing channel attention")
#         print("Input size is")
#         print(input.shape)
#         # input = networkChMap(input)
#         net = mapImage_mod_fc(
#             input, imageSize=imageSize, numFCNeurons=numFCNeurons, device=device
#         )
#         input = net(input)
#         self.add_module("map", net)
#         # batch * band *frame * c1*c2
#         input = input.permute((0, 2, 1, 3, 4))
#         # batch * frame * band * c1 * c2
#         a = nn.AvgPool3d((input.shape[2], 1, 1)).to(device)
#         m = nn.MaxPool3d((input.shape[2], 1, 1)).to(device)
#         Fa = a(input)
#         Fm = m(input)
#         # feature: batch * time * 1 * c1 * c2
#         f = torch.concat((Fa, Fm), 2)
#         # feature: batch * time * 2 * c1 * c2
#         f = f.permute((0, 2, 1, 3, 4))
#         # feature : batch * 2 * time * c1 * c2
#         # f = f.permute((0, 3, 1, 2))
#         convnet = nn.Sequential(
#             nn.Conv3d(
#                 f.shape[1], 1, (1, kernelSize[1], kernelSize[2]), padding="same",
#             ),
#             nn.ReLU(),
#             nn.BatchNorm3d(1),
#             nn.Dropout(),
#             nn.AvgPool3d((f.shape[2], 1, 1)),
#         ).to(device)
#         # feature: batch * 1 * 1 * c1 * c2
#         f = torch.sigmoid(convnet(f))
#         input = input.permute((0, 2, 1, 3, 4)) * f
#         # input : batch * band * frame * c1 * c2
#         self.add_module("avgp", a)
#         self.add_module("maxp", m)
#         self.add_module('conv', convnet)
#         print("Output size is")
#         print(input.shape)
#         # bf * band * c1 * c2

#     def forward(self, input):
#         # batch * band * frame * channel
#         # input = networkChMap(input)
#         input = self.map(input)
#         # batch * band *frame * c1*c2
#         input = input.permute((0, 2, 1, 3, 4))
#         # batch * frame * band * c1 * c2
#         Fa = self.avgp(input)
#         Fm = self.maxp(input)
#         # feature: batch * time * 1 * c1 * c2
#         f = torch.concat((Fa, Fm), 2)
#         # feature: batch * time * 2 * c1 * c2
#         f = f.permute((0, 2, 1, 3, 4))
#         # feature * batch * 2 * time * c1 * c2
#         f = torch.sigmoid(self.conv(f))
#         # feature: batch * 1 * time * c1 * c2
#         input = input.permute((0, 2, 1, 3, 4)) * f
#         # input : batch * band * frame * c1 * c2
#         return input


class mapImage(nn.Module):
    def __init__(self, input, imageSize: tuple = (9, 11)) -> None:
        super().__init__()
        self.imageSize = imageSize
        # batch * band * time * channel
        input = input.permute((0, 3, 1, 2))
        # batch * channel * band * time
        net = nn.Sequential(
            nn.Conv2d(
                in_channels=input.shape[1],
                out_channels=int(imageSize[0] * imageSize[1]),
                kernel_size=1,
                bias=False,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(int(imageSize[0] * imageSize[1])),
        )
        input = net(input)
        input = torch.reshape(
            input,
            (
                input.shape[0],
                imageSize[0],
                imageSize[1],
                input.shape[-2],
                input.shape[-1],
            ),
        )
        input = input.permute((0, 3, 4, 1, 2))
        self.add_module("net", net)

    def forward(self, input):
        input = input.permute((0, 3, 1, 2))
        input = self.net(input)
        input = torch.reshape(
            input,
            (
                input.shape[0],
                self.imageSize[0],
                self.imageSize[1],
                input.shape[-2],
                input.shape[-1],
            ),
        )
        input = input.permute((0, 3, 4, 1, 2))
        return input


class mapImage_mod(nn.Module):
    def __init__(self, input, imageSize: tuple = (16, 16), device='cpu') -> None:
        super().__init__()
        print("Initializing mapImage")
        print("Input size is")
        print(input.shape)
        self.imageSize = imageSize
        # batch * band * time * channel
        input = input.permute((0, 3, 1, 2)).unsqueeze(dim=1)
        # batch * any(1) * channel * band * time
        net = nn.Sequential(
            nn.Conv3d(
                in_channels=input.shape[1],
                out_channels=int(imageSize[0] * imageSize[1]),
                kernel_size=(input.shape[2], 1, 1),
                bias=False,
                padding='valid',
            ),
            # nn.ReLU(),
            nn.BatchNorm3d(int(imageSize[0] * imageSize[1])),
        ).to(device)
        input = net(input).squeeze()
        self.add_module("mapnet", net)
        # batch * band * time * imagesize

        input = input.view(
            input.shape[0],
            imageSize[0],
            imageSize[1],
            input.shape[-2],
            input.shape[-1],
        )
        input = input.permute((0, 3, 4, 1, 2))
        # batcg * band * time * c1 * c2
        print("Output size is")
        print(input.shape)

    def forward(self, input):
        input = input.permute((0, 3, 1, 2)).unsqueeze(dim=1)
        input = self.mapnet(input).squeeze()
        input = input.view(
            input.shape[0],
            self.imageSize[0],
            self.imageSize[1],
            input.shape[-2],
            input.shape[-1],
        )
        input = input.permute((0, 3, 4, 1, 2))
        return input
