from collections import namedtuple

import torch
from torch import nn
from torch.nn import functional as F

from torch_semantic_segmentation.nn.utils import CaptureOutput

__all__ = [
    'BiSeNet', 'BackboneOutputs',
    'bisenet_resnet18', 'bisenet_34', 'bisenet_resnet50',
]


def bisenet_resnet18(in_channels, out_channels):
    from torch_semantic_segmentation.models.resnet import resnet18

    backbone = resnet18(in_channels, 1).features
    layers = {
        '16': BackboneOutputs(backbone.layer3, 256),
        '32': BackboneOutputs(backbone.layer4, 512),
    }

    return BiSeNet(3, out_channels, backbone, layers)


def bisenet_resnet34(in_channels, out_channels):
    from torch_semantic_segmentation.models.resnet import resnet34

    backbone = resnet34(in_channels, 1).features
    layers = {
        '16': BackboneOutputs(backbone.layer3, 256),
        '32': BackboneOutputs(backbone.layer4, 512),
    }

    return BiSeNet(3, out_channels, backbone, layers)


def bisenet_resnet50(in_channels, out_channels):
    from torch_semantic_segmentation.models.resnet import resnet50

    backbone = resnet50(in_channels, 1).features
    layers = {
        '16': BackboneOutputs(backbone.layer3, 1024),
        '32': BackboneOutputs(backbone.layer4, 2048),
    }

    return BiSeNet(3, out_channels, backbone, layers)


BackboneOutputs = namedtuple('BackboneOutputs', 'layer out_channels')


class BiSeNet(nn.Module):

    def __init__(self, in_channels, out_channels, backbone, backbone_outputs):
        super(BiSeNet, self).__init__()

        self.context = backbone
        self.layers = {
            key: outputs.layer
            for key, outputs in backbone_outputs.items()
        }

        self.spatial = nn.Sequential(
            ConvBlock(in_channels, 64, kernel_size=3, padding=1, stride=2),
            ConvBlock(64, 128, kernel_size=3, padding=1, stride=2),
            ConvBlock(128, 256, kernel_size=3, padding=1, stride=2),
        )

        channels16 = backbone_outputs['16'].out_channels
        channels32 = backbone_outputs['32'].out_channels

        self.arm16 = AttentionRefinementModule(channels16, channels16)
        self.arm32 = AttentionRefinementModule(channels32, channels32)

        num_channels = 256 + channels16 + channels32
        self.ffm = FeatureFusionModule(num_channels, 128)
        self.classifier = nn.Conv2d(128, out_channels, 1)

    def forward(self, input):
        spatial = self.spatial(input)

        with CaptureOutput(self.context, self.layers) as capture:
            self.context(input)

        feat16 = capture.outputs['16']
        feat32 = capture.outputs['32']
        pool = F.adaptive_avg_pool2d(feat32, 1)

        feat16 = self.arm16(feat16)
        feat32 = self.arm32(feat32)
        feat32 = feat32 * pool

        feat32 = F.interpolate(
            feat32, size=spatial.shape[2:],
            mode='bilinear', align_corners=True)
        feat16 = F.interpolate(
            feat16, size=spatial.shape[2:],
            mode='bilinear', align_corners=True)

        context = torch.cat([feat16, feat32], dim=1)

        features = torch.cat([context, spatial], dim=1)
        features = self.ffm(features)
        logits = self.classifier(features)
        return F.interpolate(
            logits, size=input.shape[2:],
            mode='bilinear', align_corners=True)


class FeatureFusionModule(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(FeatureFusionModule, self).__init__()

        self.conv = ConvBlock(in_channels, out_channels, 3, padding=1)

        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvBlock(out_channels, out_channels, 1),
            nn.Conv2d(out_channels, out_channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, input):
        x = self.conv(input)
        attention = self.attention(x)
        return x * (1. + attention)


class AttentionRefinementModule(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(AttentionRefinementModule, self).__init__()

        if in_channels != out_channels:
            raise ValueError("input and output channels must match")

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(in_channels, out_channels, 1)
        self.activation = nn.Sigmoid()

    def forward(self, input):
        x = self.pool(input)
        x = self.conv(x)
        x = self.activation(x)
        return x * input


def ConvBlock(in_channels, out_channels, kernel_size, padding=0, stride=1, use_relu=True):
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size,
                  padding=padding, stride=stride, bias=False),
        nn.BatchNorm2d(out_channels),
    ]
    if use_relu:
        layers += [nn.ReLU(inplace=True)]
    return nn.Sequential(*layers)
