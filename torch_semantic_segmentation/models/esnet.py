from collections import OrderedDict

import torch
from torch import nn
from torch.nn import functional as F


class ESNet(nn.Sequential):

    def __init__(self, in_channels, out_channels):
        super(ESNet, self).__init__(OrderedDict([
            ('layer1', nn.Sequential(
                DownsamplingBlock(in_channels, 16),
                FCUBlock(16, 16, 3, dropout_p=0.03),
                FCUBlock(16, 16, 3, dropout_p=0.03),
                FCUBlock(16, 16, 3, dropout_p=0.03),
            )),
            ('layer2', nn.Sequential(
                DownsamplingBlock(16, 64),
                FCUBlock(64, 64, 5, dropout_p=0.03),
                FCUBlock(64, 64, 5, dropout_p=0.03),
            )),
            ('layer3', nn.Sequential(
                DownsamplingBlock(64, 128),
                FPCUBlock(128, 128, [2, 5, 9], dropout_p=0.3),
                FPCUBlock(128, 128, [2, 5, 9], dropout_p=0.3),
                FPCUBlock(128, 128, [2, 5, 9], dropout_p=0.3),
            )),
            ('layer4', nn.Sequential(
                UpsamplingBlock(128, 64),
                FCUBlock(64, 64, 5),
                FCUBlock(64, 64, 5),
            )),
            ('layer5', nn.Sequential(
                UpsamplingBlock(64, 16),
                FCUBlock(16, 16, 3),
                FCUBlock(16, 16, 3),
                FCUBlock(16, 16, 3),
            )),
            ('classifier', nn.Sequential(
                UpsamplingBlock(16, out_channels),
            ))
        ]))


class DownsamplingBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(DownsamplingBlock, self).__init__()

        if out_channels <= in_channels:
            raise ValueError(
                "output channels must be greater than the input channels")

        self.conv = nn.Conv2d(
            in_channels, out_channels - in_channels,
            kernel_size=3, padding=1, stride=2)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, input):
        x = torch.cat([
            self.conv(input),
            self.pool(input)
        ], dim=1)
        x = self.bn(x)
        return self.activation(x)


class UpsamplingBlock(nn.Sequential):

    def __init__(self, in_channels, out_channels):
        super(UpsamplingBlock, self).__init__(OrderedDict([
            ('conv', nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size=3,
                stride=2, padding=1, output_padding=1)),
            ('bn', nn.BatchNorm2d(out_channels)),
            ('activation', nn.ReLU(inplace=True)),
        ]))


class FCUBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, dropout_p=0.0):
        super(FCUBlock, self).__init__()

        if in_channels != out_channels:
            raise ValueError("input channels must match output channels")

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels,
                      kernel_size=(1, kernel_size),
                      padding=(0, kernel_size // 2)),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels,
                      kernel_size=(kernel_size, 1),
                      padding=(kernel_size // 2, 0)),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels,
                      kernel_size=(1, kernel_size),
                      padding=(0, kernel_size // 2)),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels,
                      kernel_size=(kernel_size, 1),
                      padding=(kernel_size // 2, 0)),
            nn.BatchNorm2d(in_channels)
        )

        self.activation = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(p=dropout_p)

    def forward(self, input):
        x = self.conv1(input)
        x = self.conv2(x)
        x = self.dropout(x)

        return self.activation(input + x)


class FPCUBlock(nn.Module):

    def __init__(self, in_channels, out_channels, dilations, dropout_p=0.0):
        super(FPCUBlock, self).__init__()

        if in_channels != out_channels:
            raise ValueError("input channels must match output channels")

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels,
                      kernel_size=(3, 1), padding=(1, 0)),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels,
                      kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

        self.conv2 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels,
                          kernel_size=(3, 1),
                          padding=(dilation, 0),
                          dilation=(dilation, 1)),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, in_channels,
                          kernel_size=(1, 3),
                          padding=(0, dilation),
                          dilation=(1, dilation)),
                nn.BatchNorm2d(in_channels),
            )
            for dilation in dilations
        ])

        self.activation = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(p=dropout_p)

    def forward(self, input):
        x = self.conv1(input)
        x = sum([conv(x) for conv in self.conv2])
        x = self.dropout(x)
        return self.activation(x + input)
