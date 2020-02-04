import torch
from torch import nn
from torch.nn import functional as F


class AttentionRefinementModule(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = ConvBNReLU(in_channels, out_channels,
                                kernel_size=3, padding=1)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, inputs):
        feats = self.conv1(inputs)
        atten = F.adaptive_avg_pool2d(feats, output_size=1)
        atten = self.conv2(atten)
        atten = F.sigmoid(atten)
        return torch.mul(feats, atten)


class Resnet18(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.stem = nn.Sequential(
            ConvBNReLU(in_channels, 64, kernel_size=7,
                       stride=2, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.layer1 = create_residual_layer(64, 64, repeats=2, stride=1)
        self.layer2 = create_residual_layer(64, 128, repeats=2, stride=2)
        self.layer3 = create_residual_layer(128, 256, repeats=2, stride=2)
        self.layer4 = create_residual_layer(256, 512, repeats=2, stride=2)

    def forward(self, inputs):
        x = self.stem(inputs)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return x2, x3, x4


def create_residual_layer(in_channels, out_channels, repeats=2, stride=1):
    layers = [
        ResidualBlock(in_channels, out_channels, stride=stride)
    ]
    for _ in range(1, repeats):
        layers.append(ResidualBlock(out_channels, out_channels))
    return nn.Sequential(*layers)


class ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = ConvBNReLU(
            in_channels, out_channels,
            kernel_size=3, padding=1, stride=stride)
        self.conv2 = ConvBNReLU(
            out_channels, out_channels,
            kernel_size=3, padding=1, with_relu=False)
        self.downsample = (
            ConvBNReLU(in_channels, out_channels,
                       kernel_size=1, stride=2, with_relu=False)
            if stride == 2 else None
        )

    def forward(self, inputs):
        residual = self.conv1(inputs)
        residual = self.conv2(residual)
        shortcut = (
            self.downsample(inputs)
            if self.downsample is not None
            else inputs
        )
        return F.relu(shortcut + residual)


def ConvBNReLU(in_channels, out_channels, kernel_size, stride=1, padding=0, with_relu=True):
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                  stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(out_channels),
    ]
    if with_relu:
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)
