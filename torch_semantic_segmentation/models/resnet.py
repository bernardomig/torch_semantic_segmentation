from collections import OrderedDict

import torch
from torch import nn
from torch.nn import functional as F

__all__ = [
    'ResNet',
    'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
]


def resnet18(in_channels, num_classes):
    return ResNet(in_channels, num_classes,
                  block_depth=[2, 2, 2, 2],
                  block=BasicBlock)


def resnet34(in_channels, num_classes):
    return ResNet(in_channels, num_classes,
                  block_depth=[3, 4, 6, 3],
                  block=BasicBlock)


def resnet50(in_channels, num_classes):
    return ResNet(in_channels, num_classes,
                  block_depth=[3, 4, 6, 3],
                  block=BottleneckBlock,
                  expansion=4)


def resnet101(in_channels, num_classes):
    return ResNet(in_channels, num_classes,
                  block_depth=[2, 3, 23, 3],
                  block=BottleneckBlock,
                  expansion=4)


def resnet152(in_channels, num_classes):
    return ResNet(in_channels, num_classes,
                  block_depth=[3, 8, 36, 3],
                  block=BottleneckBlock,
                  expansion=4)


class ResNet(nn.Sequential):

    def __init__(self, in_channels, num_classes, block_depth, block, expansion=1):

        features = nn.Sequential(OrderedDict([
            ('head', nn.Sequential(
                nn.Conv2d(in_channels, 64, kernel_size=7,
                          stride=2, padding=3, bias=False),
                nn.BatchNorm2d(64),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            )),
            ('layer1', make_layer(
                64, 64 * expansion, block_depth[0], block=block, stride=1)),
            ('layer2', make_layer(
                64 * expansion, 128 * expansion, block_depth[1], block=block, stride=2)),
            ('layer3', make_layer(
                128 * expansion, 256 * expansion, block_depth[2], block=block, stride=2)),
            ('layer4', make_layer(
                256 * expansion, 512 * expansion, block_depth[3], block=block, stride=2)),
        ]))

        classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512 * expansion, num_classes),
        )

        super().__init__(OrderedDict([
            ('features', features),
            ('classifier', classifier),
        ]))


def make_layer(in_channels, out_channels, num_blocks, block, stride=1):
    layers = [block(in_channels, out_channels, stride=stride)]
    for _ in range(1, num_blocks):
        layers += [block(out_channels, out_channels)]
    return nn.Sequential(*layers)


class BasicBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.conv1 = ConvBlock(
            in_channels, out_channels, kernel_size=3,
            padding=1, stride=stride)
        self.conv2 = ConvBlock(
            out_channels, out_channels, kernel_size=3,
            padding=1, use_relu=False)

        self.downsample = (
            ConvBlock(
                in_channels, out_channels, kernel_size=1,
                stride=stride, use_relu=False)
            if in_channels != out_channels or stride == 2 else None)

    def forward(self, input):
        x = self.conv1(input)
        x = self.conv2(x)
        s = (input
             if self.downsample is None
             else self.downsample(input))
        return F.relu(x + s)


class BottleneckBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, expansion=4):
        super().__init__()

        width = out_channels // expansion

        self.conv1 = ConvBlock(
            in_channels, width, kernel_size=1)
        self.conv2 = ConvBlock(
            width, width, kernel_size=3,
            padding=1, stride=stride)
        self.conv3 = ConvBlock(
            width, out_channels,
            kernel_size=1, use_relu=False)

        self.downsample = (
            ConvBlock(
                in_channels, out_channels, kernel_size=1,
                stride=stride, use_relu=False)
            if in_channels != out_channels or stride == 2 else None)

    def forward(self, input):
        x = self.conv1(input)
        x = self.conv2(x)
        x = self.conv3(x)
        s = (input
             if self.downsample is None
             else self.downsample(input))
        return F.relu(x + s)


def ConvBlock(in_channels, out_channels, kernel_size,
              padding=0, stride=1,
              use_relu=True):
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size,
                  padding=padding, stride=stride, bias=False),
        nn.BatchNorm2d(out_channels),
    ]
    if use_relu:
        layers += [nn.ReLU(inplace=True)]

    return nn.Sequential(*layers)
