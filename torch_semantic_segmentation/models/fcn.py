from collections import namedtuple

import torch
from torch import nn
from torch.nn import functional as F

from torch_semantic_segmentation.nn.utils import CaptureOutput


def fcn32_resnet18(in_channels, out_channels):
    from torch_semantic_segmentation.models.resnet import resnet18

    backbone = resnet18(in_channels, 1).features
    return FCN32(in_channels, out_channels, backbone, 512)


def fcn32_resnet34(in_channels, out_channels):
    from torch_semantic_segmentation.models.resnet import resnet34

    backbone = resnet34(in_channels, 1).features
    return FCN32(in_channels, out_channels, backbone, 512)


def fcn16_resnet18(in_channels, out_channels):
    from torch_semantic_segmentation.models.resnet import resnet18

    backbone = resnet18(in_channels, 1).features
    backbone_outputs = {
        '16': BackboneOutputs(backbone.layer3, 256),
        '32': BackboneOutputs(backbone.layer4, 512),
    }

    return FCN16(in_channels, out_channels, backbone, backbone_outputs)


def fcn8_resnet18(in_channels, out_channels):
    from torch_semantic_segmentation.models.resnet import resnet18

    backbone = resnet18(in_channels, 1).features
    backbone_outputs = {
        '8': BackboneOutputs(backbone.layer2, 128),
        '16': BackboneOutputs(backbone.layer3, 256),
        '32': BackboneOutputs(backbone.layer4, 512),
    }

    return FCN8(in_channels, out_channels, backbone, backbone_outputs)


BackboneOutputs = namedtuple('BackboneOutputs', 'layer out_channels')


class FCN32(nn.Module):

    def __init__(self, in_channels, out_channels, backbone, backbone_out_channels):
        super(FCN32, self).__init__()

        self.backbone = backbone
        self.classifier = nn.Conv2d(
            backbone_out_channels, out_channels, 1)

    def forward(self, input):
        feat = self.backbone(input)
        x = self.classifier(feat)
        return F.interpolate(
            x, scale_factor=32,
            mode='bilinear', align_corners=True)


class FCN16(nn.Module):
    def __init__(self, in_channels, out_channels, backbone, backbone_outputs):
        super(FCN16, self).__init__()

        self.backbone = backbone

        self.classifiers = nn.ModuleDict({
            key: nn.Conv2d(outputs.out_channels, out_channels, 1)
            for key, outputs in backbone_outputs.items()
        })

        self.layers = {
            key: outputs.layer
            for key, outputs in backbone_outputs.items()
        }

    def forward(self, input):
        with CaptureOutput(self.backbone, self.layers) as capture:
            self.backbone(input)

        feat16 = capture.outputs['16']
        feat32 = capture.outputs['32']

        pred16 = self.classifiers['16'](feat16)
        pred32 = self.classifiers['32'](feat32)
        pred32 = F.interpolate(
            pred32, scale_factor=2,
            mode='bilinear', align_corners=True)

        pred = pred16 + pred32
        return F.interpolate(
            pred, scale_factor=16,
            mode='bilinear', align_corners=True)


class FCN8(nn.Module):

    def __init__(self, in_channels, out_channels, backbone, backbone_outputs):
        super(FCN8, self).__init__()

        self.backbone = backbone

        self.classifiers = nn.ModuleDict({
            key: nn.Conv2d(outputs.out_channels, out_channels, 1)
            for key, outputs in backbone_outputs.items()
        })

        self.layers = {
            key: outputs.layer
            for key, outputs in backbone_outputs.items()
        }

    def forward(self, input):
        with CaptureOutput(self.backbone, self.layers) as capture:
            self.backbone(input)
        feat8 = capture.outputs['8']
        feat16 = capture.outputs['16']
        feat32 = capture.outputs['32']

        pred8 = self.classifiers['8'](feat8)
        pred16 = self.classifiers['16'](feat16)
        pred32 = self.classifiers['32'](feat32)

        pred32 = F.interpolate(
            pred32, scale_factor=2,
            mode='bilinear')
        pred16 = pred16 + pred32

        pred16 = F.interpolate(
            pred16, scale_factor=2,
            mode='bilinear')
        pred = pred8 + pred16

        return F.interpolate(
            pred, scale_factor=8,
            mode='bilinear')
