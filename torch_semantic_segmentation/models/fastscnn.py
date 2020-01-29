import torch
from torch import nn
from torch.nn import functional as F


class FastSCNN(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        # The Learning to Downsample module
        # It encodes low level features in a efficient way
        # It is composed by three convolutional layers, where the first one is
        # a regular conv and the other two are depthwise separable conv
        # layers.
        # The first convolutional layer is a regular conv because there is no
        # advantage in using a ds conv in such small number of channels.
        # All layers are a spatial kernel of 3x3 and have a stride of 2 for a total downsample of 8 times.
        # Also, there is no nonlinearity between the depthwise and pointwise conv.
        self.downsample = nn.Sequential(
            Conv2dBlock(in_channels, 32, kernel_size=3, padding=1, stride=2),
            DSConv2dBlock(32, 48, kernel_size=3, padding=1, stride=2),
            DSConv2dBlock(48, 64, kernel_size=3, padding=1, stride=2),
        )

        # The Global Feature Extractor module is aimed at capturing the global
        # context for the task of image segmentation.
        # This module directly takes the 1/8 downsampled output of the
        # Learning to Downsample module, performs a feature encoding using the
        # MobileNet bottleneck residual block and then performs a pyramid pooling
        # at the end to aggregate the different region-based context information.
        self.features = nn.Sequential(
            BottleneckModule(64, 64, expansion=6, repeats=3, stride=2),
            BottleneckModule(64, 96, expansion=6, repeats=3, stride=2),
            BottleneckModule(96, 128, expansion=6, repeats=3, stride=1),
            PyramidPoolingModule(128, 128)
        )

        # The Feature Fusion adds the low-resolution features from the
        # Global Feature Encoder and the high-resolution features from the
        # Learning to Downsample Module.
        self.fusion = FeatureFusionModule((128, 64), 128, scale_factor=4)

        # The classifier discriminates the classes from the features produced
        # by fusion module.
        self.classifier = Classifier(128, out_channels)

        # It is helpful the use addicional losses from the Learning to Downsample
        # and the Global Feature Extractor. This auxiliary classifiers are only used
        # during training.
        self.aux_classifier = nn.ModuleDict({
            'downsample': Classifier(64, out_channels),
            'features': Classifier(128, out_channels),
        })

    def forward(self, input):
        downsample = self.downsample(input)
        features = self.features(downsample)
        fusion = self.fusion(features, downsample)
        classes = self.classifier(fusion)

        if self.training:
            aux_1 = self.aux_classifier['downsample'](downsample)
            aux_2 = self.aux_classifier['features'](features)

            return (
                F.interpolate(classes, scale_factor=8,
                              mode='bilinear', align_corners=True),
                F.interpolate(aux_1, scale_factor=8,
                              mode='bilinear', align_corners=True),
                F.interpolate(aux_2, scale_factor=4 * 8,
                              mode='bilinear', align_corners=True),
            )
        else:
            return F.interpolate(
                classes, scale_factor=8, mode='bilinear', align_corners=True)


class FeatureFusionModule(nn.Module):

    def __init__(self, in_channels, out_channels, scale_factor):
        super().__init__()
        lowres_channels, highres_channels = in_channels

        self.lowres = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=scale_factor),
            DWConv2dBlock(lowres_channels, lowres_channels, kernel_size=3,
                          padding=scale_factor, dilation=scale_factor),
            Conv2dBlock(lowres_channels, out_channels,
                        kernel_size=1, use_activation=False),
        )

        self.highres = nn.Sequential(
            Conv2dBlock(highres_channels, out_channels,
                        kernel_size=1, use_activation=False)
        )

    def forward(self, lowres, highres):
        lowres = self.lowres(lowres)
        highres = self.highres(highres)
        return F.relu(lowres + highres)


def Classifier(in_channels, out_channels):
    return nn.Sequential(
        DSConv2dBlock(in_channels, in_channels, kernel_size=3, padding=1),
        DSConv2dBlock(in_channels, in_channels, kernel_size=3, padding=1),
        nn.Dropout(0.1),
        nn.Conv2d(in_channels, out_channels, kernel_size=1),
    )


class PyramidPoolingModule(nn.Module):

    def __init__(self, in_channels, out_channels, pyramids=(1, 2, 3, 6)):
        super().__init__()

        self.pyramids = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(bin),
                Conv2dBlock(in_channels, in_channels //
                            len(pyramids), kernel_size=1),
            )
            for bin in pyramids
        ])

        self.conv = Conv2dBlock(in_channels * 2, out_channels, kernel_size=1)

    def forward(self, input):
        pools = [
            F.interpolate(
                pool(input), size=input.shape[2:], mode='bilinear', align_corners=True)
            for pool in self.pyramids.children()]
        x = torch.cat([input, *pools], dim=1)
        return self.conv(x)


def BottleneckModule(in_channels, out_channels, expansion, repeats=1, stride=1):
    layers = [
        BottleneckBlock(in_channels, out_channels,
                        expansion=expansion, stride=stride)
    ]
    for _ in range(1, repeats):
        layers.append(
            BottleneckBlock(out_channels, out_channels, expansion=expansion)
        )
    return nn.Sequential(*layers)


class BottleneckBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, expansion=6):
        super().__init__()

        expansion_channels = expansion * in_channels
        self.conv1 = Conv2dBlock(
            in_channels, expansion_channels, kernel_size=1)
        self.conv2 = DWConv2dBlock(
            expansion_channels, expansion_channels, kernel_size=3,
            padding=1, stride=stride)
        self.conv3 = Conv2dBlock(
            expansion_channels, out_channels, kernel_size=1, use_activation=False)

    def forward(self, input: torch.Tensor):

        x = self.conv1(input)
        x = self.conv2(x)
        x = self.conv3(x)

        if x.shape == input.shape:
            x = x + input

        return F.relu(x)


def Conv2dBlock(in_channels, out_channels, kernel_size,
                stride=1, padding=0, dilation=1, use_activation=True):
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size,
                  stride=stride, padding=padding, dilation=dilation, bias=False),
        nn.BatchNorm2d(out_channels),
    ]
    if use_activation:
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


def DWConv2dBlock(in_channels, out_channels, kernel_size,
                  stride=1, padding=0, dilation=1, use_activation=True):
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size,
                  stride=stride, padding=padding, dilation=dilation, bias=False, groups=in_channels),
        nn.BatchNorm2d(out_channels),
    ]
    if use_activation:
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


def DSConv2dBlock(in_channels, out_channels, kernel_size,
                  stride=1, padding=0, dilation=1, use_activation=True):
    layers = [
        nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride,
                  padding=padding, dilation=dilation, groups=in_channels, bias=False),
        nn.BatchNorm2d(in_channels),
        nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
        nn.BatchNorm2d(out_channels),
    ]
    if use_activation:
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)
