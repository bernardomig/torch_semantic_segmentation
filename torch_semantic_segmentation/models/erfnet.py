from collections import OrderedDict

import torch
from torch import nn
from torch.nn import functional as F

__all__ = [
    'ERFNet',
    'erfnet',
]


def erfnet(in_channels, out_channels):
    return ERFNet(in_channels, out_channels)


class ERFNet(nn.Sequential):

    def __init__(self, in_channels, out_channels):
        super(ERFNet, self).__init__(OrderedDict([
            ('layer1', nn.Sequential(
                DownsamplingBlock(in_channels, 16),
            )),
            ('layer2', nn.Sequential(
                DownsamplingBlock(16, 64),
                NonBottleneck1DBlock(64, 64, dropout_p=0.03),
                NonBottleneck1DBlock(64, 64, dropout_p=0.03),
                NonBottleneck1DBlock(64, 64, dropout_p=0.03),
                NonBottleneck1DBlock(64, 64, dropout_p=0.03),
                NonBottleneck1DBlock(64, 64, dropout_p=0.03),
            )),
            ('layer3', nn.Sequential(
                DownsamplingBlock(64, 128),
                NonBottleneck1DBlock(128, 128, dilation=2, dropout_p=0.3),
                NonBottleneck1DBlock(128, 128, dilation=4, dropout_p=0.3),
                NonBottleneck1DBlock(128, 128, dilation=8, dropout_p=0.3),
                NonBottleneck1DBlock(128, 128, dilation=16, dropout_p=0.3),
                NonBottleneck1DBlock(128, 128, dilation=2, dropout_p=0.3),
                NonBottleneck1DBlock(128, 128, dilation=4, dropout_p=0.3),
                NonBottleneck1DBlock(128, 128, dilation=8, dropout_p=0.3),
                NonBottleneck1DBlock(128, 128, dilation=16, dropout_p=0.3),
            )),
            ('layer4', nn.Sequential(
                UpsamplingBlock(128, 64),
                NonBottleneck1DBlock(64, 64, dropout_p=0.3),
                NonBottleneck1DBlock(64, 64, dropout_p=0.3),
            )),
            ('layer5', nn.Sequential(
                UpsamplingBlock(64, 16),
                NonBottleneck1DBlock(16, 16, dropout_p=0.3),
                NonBottleneck1DBlock(16, 16, dropout_p=0.3),
            )),
            ('classifier', nn.Conv2d(16, out_channels, 1)),
        ]))


class NonBottleneck1DBlock(nn.Module):

    def __init__(self, in_channels, out_channels, dilation=1, dropout_p=0.):
        super(NonBottleneck1DBlock, self).__init__()

        if in_channels != out_channels:
            raise ValueError("input and output channels must match")

        self.conv1 = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels, (3, 1),
                          padding=(1, 0), bias=False),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels, (1, 3),
                          padding=(0, 1), bias=False),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
            ),
        )

        self.conv2 = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels, (3, 1),
                          padding=(dilation, 0),
                          dilation=(dilation, 0),
                          bias=False),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels, (1, 3),
                          padding=(0, dilation),
                          dilation=(0, dilation),
                          bias=False),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
            ),
        )

        self.activation = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(p=dropout_p)

    def forward(self, input):
        x = self.conv1(input)
        x = self.conv2(x)
        x = self.dropout(x)
        return self.activation(input + x)


class DownsamplingBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(DownsamplingBlock, self).__init__()

        if out_channels <= in_channels:
            raise ValueError(
                "output channels must be greater than the input channels")

        self.conv = nn.Conv2d(in_channels, out_channels - in_channels,
                              kernel_size=3, padding=1, stride=2, bias=False)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, input):
        x = torch.cat([
            self.conv(input),
            self.pool(input),
        ], dim=1)
        x = self.bn(x)
        x = self.activation(x)
        return x


class UpsamplingBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(UpsamplingBlock, self).__init__(OrderedDict([
            ('conv', nn.ConvTranspose2d(
                in_channels, out_channels, 3,
                padding=1, output_padding=1, bias=False)),
            ('bn', nn.BatchNorm2d(out_channels)),
            ('activation', nn.ReLU(inplace=True)),
        ]))
