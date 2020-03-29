import torch
from torch import nn
from torch.nn import functional as F

__all__ = [
    'ContextNet',
    'contextnet12',
    'contextnet14',
    'contextnet18',
]


def contextnet12(in_channels, out_channels):
    return ContextNet(in_channels, out_channels,
                      scale_factor=2)


def contextnet14(in_channels, out_channels):
    return ContextNet(in_channels, out_channels,
                      scale_factor=4)


def contextnet18(in_channels, out_channels):
    return ContextNet(in_channels, out_channels,
                      scale_factor=8)


class ContextNet(nn.Module):

    scale_factor: int = 4

    def __init__(self, in_channels, out_channels, scale_factor=4):
        super(ContextNet, self).__init__()

        self.scale_factor = scale_factor

        self.spatial = nn.Sequential(
            ConvBlock(in_channels, 32, 3, padding=1, stride=2),
            DWConvBlock(32, 32, kernel_size=3, padding=1, stride=2),
            ConvBlock(32, 64, 1),
            DWConvBlock(64, 64, kernel_size=3, padding=1, stride=2),
            ConvBlock(64, 128, 1),
            DWConvBlock(128, 128, kernel_size=3, padding=1, stride=1),
            ConvBlock(128, 128, 1),
        )

        self.context = nn.Sequential(
            ConvBlock(in_channels, 32, 3, padding=1, stride=2),
            BottleneckBlock(32, 32, expansion=1),
            BottleneckBlock(32, 32, expansion=6),
            LinearBottleneck(32, 48, 3, stride=2),
            LinearBottleneck(48, 64, 3, stride=2),
            LinearBottleneck(64, 96, 2),
            LinearBottleneck(96, 128, 2),
            ConvBlock(128, 128, 3, padding=1),
        )

        self.feature_fusion = FeatureFusionModule((128, 128), 128)

        self.classifier = Classifier(128, out_channels)

    def forward(self, input):
        spatial = self.spatial(input)

        context = F.interpolate(
            input, scale_factor=1 / self.scale_factor,
            mode='bilinear', align_corners=True)
        context = self.context(context)

        fusion = self.feature_fusion(context, spatial)

        classes = self.classifier(fusion)

        return F.interpolate(
            classes, scale_factor=8,
            mode='bilinear', align_corners=True)


def Classifier(in_channels, out_channels):
    return nn.Sequential(
        DWConvBlock(in_channels, in_channels, 3, padding=1),
        ConvBlock(in_channels, in_channels, 1),
        DWConvBlock(in_channels, in_channels, 3, padding=1),
        ConvBlock(in_channels, in_channels, 1),
        nn.Dropout(p=0.1),
        nn.Conv2d(in_channels, out_channels, 1),
    )


def LinearBottleneck(in_channels, out_channels, num_blocks, expansion=6, stride=1):
    layers = [
        BottleneckBlock(
            in_channels, out_channels,
            stride=stride, expansion=expansion)]

    for _ in range(1, num_blocks):
        layers += [
            BottleneckBlock(
                out_channels, out_channels, expansion=expansion)
        ]
    return nn.Sequential(*layers)


class FeatureFusionModule(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(FeatureFusionModule, self).__init__()

        lowres_channels, highres_channels = in_channels
        self.lowres = nn.Sequential(
            DWConvBlock(lowres_channels, lowres_channels,
                        kernel_size=3, padding=4, dilation=4),
            ConvBlock(lowres_channels, out_channels, 1, use_relu=False)
        )
        self.highres = ConvBlock(
            highres_channels, out_channels, 1, use_relu=False)

    def forward(self, lowres, highres):
        lowres = F.interpolate(
            lowres, size=highres.shape[2:],
            mode='bilinear', align_corners=True)
        lowres = self.lowres(lowres)

        highres = self.highres(highres)

        return F.relu(lowres + highres)


class BottleneckBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, expansion=6):
        super(BottleneckBlock, self).__init__()

        expansion_channels = in_channels * expansion
        self.conv1 = ConvBlock(in_channels, expansion_channels, 1)
        self.conv2 = DWConvBlock(
            expansion_channels, expansion_channels, 3,
            padding=1, stride=stride)
        self.conv3 = ConvBlock(
            expansion_channels, out_channels, 1, use_relu=False)

    def forward(self, input):
        x = self.conv1(input)
        x = self.conv2(x)
        x = self.conv3(x)
        if x.shape == input.shape:
            x = input + x
        return F.relu(x)


def DWConvBlock(in_channels, out_channels, kernel_size,
                padding=0, stride=1, dilation=1,
                use_relu=True):
    if in_channels != out_channels:
        raise ValueError(
            "input and output channels must be the same in depthwise convolution")

    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size,
                  padding=padding, stride=stride, dilation=dilation,
                  groups=in_channels, bias=False),
        nn.BatchNorm2d(out_channels),
    ]
    if use_relu:
        layers += [nn.ReLU(inplace=True)]
    return nn.Sequential(*layers)


def ConvBlock(in_channels, out_channels, kernel_size, padding=0, stride=1, use_relu=True):
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size,
                  padding=padding, stride=stride,
                  bias=False),
        nn.BatchNorm2d(out_channels),
    ]
    if use_relu:
        layers += [nn.ReLU(inplace=True)]
    return nn.Sequential(*layers)
