from collections import OrderedDict
import torch
from torch import nn
from torch.nn import functional as F

__all__ = [
    'FCDenseNet',
    'fc_densenet_57', 'fc_densenet_67', 'fc_densenet_103',
]


def fc_densenet_57(in_channels, out_channels):
    return FCDenseNet(
        in_channels, out_channels,
        initial_channels=48,
        block_depth=[4, 4, 4, 4, 4],
        bottleneck_depth=4,
        growth=12,
    )


def fc_densenet_67(in_channels, out_channels):
    return FCDenseNet(
        in_channels, out_channels,
        initial_channels=48,
        block_depth=[5, 5, 5, 5, 5],
        bottleneck_depth=5,
        growth=16,
    )


def fc_densenet_103(in_channels, out_channels):
    return FCDenseNet(
        in_channels, out_channels,
        initial_channels=48,
        block_depth=[4, 5, 7, 10, 12],
        bottleneck_depth=15,
        growth=16,
    )


class FCDenseNet(nn.Module):
    def __init__(self, in_channels, out_channels,
                 initial_channels=48,
                 block_depth=[4, 5, 7, 10, 12],
                 growth=16,
                 bottleneck_depth=15,
                 ):
        super(FCDenseNet, self).__init__()

        self.head = nn.Conv2d(in_channels, initial_channels,
                              kernel_size=3, padding=1, bias=False)

        channels = [initial_channels]

        downpath = []
        for depth in block_depth:
            channels += [channels[-1] + depth * growth]
            layer = nn.ModuleDict({
                'block': DenseBlock(channels[-2], channels[-1], growth, depth),
                'transition': TransitionDownBlock(channels[-1], channels[-1]),
            })
            prev_channels = channels
            downpath += [layer]
        self.downpath = nn.ModuleList(downpath)

        self.bottleneck = DenseBlock(channels[-1], growth * bottleneck_depth,
                                     growth=growth,
                                     num_blocks=bottleneck_depth,
                                     use_skip=False)
        prev_channels = growth * bottleneck_depth

        uppath = []
        for depth in reversed(block_depth):
            skip_channels = channels.pop()
            layer = nn.ModuleDict({
                'transition': TransitionUpBlock(prev_channels, prev_channels),
                'block': DenseBlock(
                    prev_channels + skip_channels,
                    growth * depth,
                    growth, depth,
                    use_skip=False),
            })
            prev_channels = growth * depth
            uppath += [layer]
        self.uppath = nn.ModuleList(uppath)

        self.classifier = nn.Conv2d(prev_channels, out_channels, 1)

    def forward(self, input):
        x = self.head(input)

        skip = []
        for layer in self.downpath.children():
            x = layer['block'](x)
            skip.append(x)
            x = layer['transition'](x)

        x = self.bottleneck(x)

        for layer in self.uppath.children():
            x = layer['transition'](x)
            s = skip.pop()
            x = torch.cat([x, s], dim=1)
            x = layer['block'](x)

        return self.classifier(x)


class DenseBlock(nn.ModuleList):

    def __init__(self, in_channels, out_channels, growth=12, num_blocks=4, use_skip=True):
        expected_output = (in_channels if use_skip else 0) + \
            num_blocks * growth
        if expected_output != out_channels:
            raise ValueError(
                "output channels must be {}. Got {}."
                .format(expected_output, out_channels)
            )

        layers = []
        for i in range(num_blocks):
            layers += [
                ConvBlock(in_channels + i * growth, growth, 3, 1),
            ]

        super(DenseBlock, self).__init__(layers)

        self.use_skip = use_skip

    def forward(self, input):
        x = input
        for layer in self.children():
            feat = layer(x)
            x = torch.cat([x, feat], dim=1)
        return x if self.use_skip else x[:, input.size(1):]


class TransitionDownBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(TransitionDownBlock, self).__init__(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.Dropout(p=0.2),
            nn.MaxPool2d(kernel_size=2),
        )


class TransitionUpBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(TransitionUpBlock, self).__init__(
            nn.ConvTranspose2d(in_channels, out_channels, 3,
                               stride=2, padding=1, output_padding=1),
        )


def ConvBlock(in_channels, out_channels, kernel_size,
              padding=0):
    return nn.Sequential(
        nn.BatchNorm2d(in_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels, out_channels, kernel_size,
                  padding=padding, bias=False),
        nn.Dropout(p=0.2),
    )
