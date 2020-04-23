import torch
from torch import nn
from torch.nn import functional as F


class PSPNet(nn.Module):

    def __init__(self, backbone, out_channels,
                 feature_channels):
        super().__init__()

        self.backbone = backbone

        self.ppm = PyramidPoolingModule(
            feature_channels, feature_channels, pools=[1, 2, 3, 6])

        self.classifier = nn.Conv2d(feature_channels * 2, out_channels, 1)

    def forward(self, input):
        feat = self.backbone(input)
        pools = self.ppm(feat)
        x = torch.cat([feat, pools], dim=1)
        return self.classifier(x)


class PyramidPoolingModule(nn.ModuleList):

    def __init__(self, in_channels, out_channels, pools=[1, 2, 3, 6]):

        if out_channels % len(pools) != 0:
            raise ValueError(
                "output channels must be divisible by the number of pools")

        pool_channels = out_channels // len(pools)

        pools = [
            nn.Sequential(
                nn.AdaptiveAvgPool2d(pool_size),
                nn.Sequential(
                    nn.Conv2d(in_channels, pool_channels, 1, bias=False),
                    nn.BatchNorm2d(pool_channels),
                    nn.ReLU(inplace=True),
                ),
            )
            for pool_size in pools
        ]

        super(PyramidPoolingModule, self).__init__(pools)

    def forward(self, input):
        size = input.shape[2:]

        pools = [pool(input) for pool in self.children()]
        pools = torch.cat([
            F.interpolate(pool, size=size, mode='bilinear', align_corners=True)
            for pool in pools
        ], dim=1)
        return pools
