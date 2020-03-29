from typing import Tuple

import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F

__all__ = ['ENet', 'enet']


def enet(in_channels, out_channels):
    return ENet(in_channels, out_channels)


class ENet(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.head = InitialBlock(in_channels, 16)

        self.layer1 = nn.ModuleList([
            DownsamplingBlock(16, 64, dropout_p=0.01),
            RegularBlock(64, 64, dropout_p=0.01),
            RegularBlock(64, 64, dropout_p=0.01),
            RegularBlock(64, 64, dropout_p=0.01),
            RegularBlock(64, 64, dropout_p=0.01),
        ])

        self.layer2 = nn.ModuleList([
            DownsamplingBlock(64, 128),
            RegularBlock(128, 128),
            RegularBlock(128, 128, dilation=2),
            RegularBlock(128, 128, kernel_size=5),
            RegularBlock(128, 128, dilation=4),
            RegularBlock(128, 128),
            RegularBlock(128, 128, dilation=8),
            RegularBlock(128, 128, kernel_size=5),
            RegularBlock(128, 128, dilation=16),
        ])

        self.layer3 = nn.ModuleList([
            RegularBlock(128, 128),
            RegularBlock(128, 128, dilation=2),
            RegularBlock(128, 128, kernel_size=5),
            RegularBlock(128, 128, dilation=4),
            RegularBlock(128, 128),
            RegularBlock(128, 128, dilation=8),
            RegularBlock(128, 128, kernel_size=5),
            RegularBlock(128, 128, dilation=16),
        ])

        self.layer4 = nn.ModuleList([
            UpsamplingBlock(128, 64),
            RegularBlock(64, 64),
            RegularBlock(64, 64),
        ])

        self.layer5 = nn.ModuleList([
            UpsamplingBlock(64, 16),
            RegularBlock(16, 16),
        ])

        self.classifier = nn.Conv2d(16, out_channels, 3, padding=1)

    def forward(self, input):
        x = self.head(input)
        indices = []
        layers = [self.layer1, self.layer2, self.layer3,
                  self.layer4, self.layer5]
        for layer in layers:
            for block in layer.children():
                if isinstance(block, RegularBlock):
                    x = block(x)
                elif isinstance(block, DownsamplingBlock):
                    x, i = block(x)
                    indices = [i, *indices]
                elif isinstance(block, UpsamplingBlock):
                    i, *indices = indices
                    x = block(x, i)
                else:
                    raise RuntimeError("Unknown block type in ENet")

        x = self.classifier(x)

        return F.interpolate(x, scale_factor=2,
                             mode='bilinear', align_corners=True)


class InitialBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        if out_channels <= in_channels:
            raise ValueError(
                "output channels must be greater than the input channels")

        super().__init__()

        self.conv = ConvBlock(
            in_channels, out_channels - in_channels,
            kernel_size=3, padding=1, stride=2, with_relu=False)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.activation = nn.PReLU()

    def forward(self, input):
        left = self.conv(input)
        right = self.pool(input)
        x = torch.cat([left, right], dim=1)
        return self.activation(x)


class RegularBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, dilation=1, projection_ratio=4, dropout_p=0.1):
        super().__init__()

        width = in_channels // projection_ratio
        self.conv1 = ConvBlock(in_channels, width, 1)

        if kernel_size == 3:
            self.conv2 = ConvBlock(
                width, width, 3,
                padding=dilation, dilation=dilation)
        elif kernel_size == 5:
            self.conv2 = nn.Sequential(
                ConvBlock(width, width, (1, 5),
                          padding=(0, 2), with_relu=False),
                ConvBlock(width, width, (5, 1),
                          padding=(2, 0), with_relu=False),
                nn.PReLU(),
            )
        else:
            raise ValueError(
                "kernel_size must be either 3 or 5. Got {}."
                .format(kernel_size))

        self.conv3 = ConvBlock(width, out_channels, 1, with_relu=False)

        self.dropout = nn.Dropout2d(p=dropout_p)
        self.activation = nn.PReLU()

    def forward(self, input):
        x = self.conv1(input)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.dropout(x)

        if input.shape != x.shape:
            input = pad_zeros(input, x.shape)

        return self.activation(x + input)


class DownsamplingBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                 projection_ratio=4, dropout_p=0.1):
        super().__init__()

        width = in_channels // projection_ratio

        self.conv1 = ConvBlock(in_channels, width, 1)
        self.conv2 = ConvBlock(width, width, 3, 1, stride=2)
        self.conv3 = ConvBlock(width, out_channels, 1, with_relu=False)
        self.dropout = nn.Dropout2d(p=dropout_p)

        self.downsample = nn.MaxPool2d(kernel_size=2, return_indices=True)

        self.activation = nn.PReLU()

    def forward(self, input):
        x = self.conv1(input)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.dropout(x)

        input, indices = self.downsample(input)

        if input.shape != x.shape:
            input = pad_zeros(input, x.shape)

        return self.activation(x + input), indices


class UpsamplingBlock(nn.Module):

    def __init__(self, in_channels, out_channels, projection_ratio=4, dropout_p=0.1):
        super().__init__()

        width = in_channels // projection_ratio

        self.conv1 = ConvBlock(in_channels, width, 1)
        self.conv2 = nn.Sequential(
            nn.ConvTranspose2d(width, width, 3,
                               stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.PReLU(),
        )
        self.conv3 = ConvBlock(width, out_channels, 1)
        self.dropout = nn.Dropout2d(p=dropout_p)

        self.upsample = nn.ModuleDict({
            'unpool': nn.MaxUnpool2d(kernel_size=2, stride=2),
            'conv': ConvBlock(in_channels, out_channels, 1, with_relu=False),
        })

        self.activation = nn.PReLU()

    def forward(self, input, indices):
        left = self.conv1(input)
        left = self.conv2(left)
        left = self.conv3(left)
        left = self.dropout(left)

        right = self.upsample['conv'](input)
        right = self.upsample['unpool'](
            right, indices=indices, output_size=left.size())

        return self.activation(left + right)


def ConvBlock(in_channels, out_channels, kernel_size,
              padding=0, stride=1, dilation=1, with_relu=True):
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size,
                  padding=padding, stride=stride, dilation=dilation,
                  bias=False),
        nn.BatchNorm2d(out_channels),
    ]
    if with_relu:
        layers += [nn.PReLU()]
    return nn.Sequential(*layers)


@torch.jit.script
def pad_zeros(input: Tensor, shape: Tuple[int, int, int, int]):
    b, _, h, w = input.shape

    c = int(shape[1] - input.size(1))

    pad = torch.zeros((b, c, h, w)).to(input)
    return torch.cat([input, pad], dim=1)
