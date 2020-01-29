import torch
from torch import nn
from torch.nn import functional as F


class ENet(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.init_block = InitialBlock(3, 16)

        self.bottleneck1_0 = DownsamplingBottleneck(16, 64, dropout_p=0.01)
        self.bottleneck1_1 = RegularBottleneck(64, 64, dropout_p=0.01)
        self.bottleneck1_2 = RegularBottleneck(64, 64, dropout_p=0.01)
        self.bottleneck1_3 = RegularBottleneck(64, 64, dropout_p=0.01)
        self.bottleneck1_4 = RegularBottleneck(64, 64, dropout_p=0.01)

        self.bottleneck2_0 = DownsamplingBottleneck(64, 128)
        self.bottleneck2_1 = RegularBottleneck(128, 128)
        self.bottleneck2_2 = RegularBottleneck(128, 128, dilation=2)
        self.bottleneck2_3 = RegularBottleneck(128, 128, kernel_size=5)
        self.bottleneck2_4 = RegularBottleneck(128, 128, dilation=4)
        self.bottleneck2_5 = RegularBottleneck(128, 128)
        self.bottleneck2_6 = RegularBottleneck(128, 128, dilation=8)
        self.bottleneck2_7 = RegularBottleneck(128, 128, kernel_size=5)
        self.bottleneck2_8 = RegularBottleneck(128, 128, dilation=16)

        self.bottleneck3_1 = RegularBottleneck(128, 128)
        self.bottleneck3_2 = RegularBottleneck(128, 128, dilation=2)
        self.bottleneck3_3 = RegularBottleneck(128, 128, kernel_size=5)
        self.bottleneck3_4 = RegularBottleneck(128, 128, dilation=4)
        self.bottleneck3_5 = RegularBottleneck(128, 128)
        self.bottleneck3_6 = RegularBottleneck(128, 128, dilation=8)
        self.bottleneck3_7 = RegularBottleneck(128, 128, kernel_size=5)
        self.bottleneck3_8 = RegularBottleneck(128, 128, dilation=16)

        self.bottleneck4_0 = UpsamplingBottleneck(128, 64)
        self.bottleneck4_1 = RegularBottleneck(64, 64)
        self.bottleneck4_2 = RegularBottleneck(64, 64)

        self.bottleneck5_0 = UpsamplingBottleneck(64, 16)
        self.bottleneck5_1 = RegularBottleneck(16, 16)

        self.fullconv = nn.Conv2d(16, out_channels, kernel_size=3, padding=1)

    def forward(self, x):

        x = self.init_block(x)

        x, i1 = self.bottleneck1_0(x)

        x = self.bottleneck1_1(x)
        x = self.bottleneck1_2(x)
        x = self.bottleneck1_3(x)
        x = self.bottleneck1_4(x)

        x, i2 = self.bottleneck2_0(x)
        x = self.bottleneck2_1(x)
        x = self.bottleneck2_2(x)
        x = self.bottleneck2_3(x)
        x = self.bottleneck2_4(x)
        x = self.bottleneck2_5(x)
        x = self.bottleneck2_6(x)
        x = self.bottleneck2_7(x)
        x = self.bottleneck2_8(x)

        x = self.bottleneck3_1(x)
        x = self.bottleneck3_2(x)
        x = self.bottleneck3_3(x)
        x = self.bottleneck3_4(x)
        x = self.bottleneck3_5(x)
        x = self.bottleneck3_6(x)
        x = self.bottleneck3_7(x)
        x = self.bottleneck3_8(x)

        x = self.bottleneck4_0(x, i2)
        x = self.bottleneck4_1(x)
        x = self.bottleneck4_2(x)

        x = self.bottleneck5_0(x, i1)
        x = self.bottleneck5_1(x)

        x = self.fullconv(x)

        return F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)


class InitialBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        if out_channels <= in_channels:
            raise ValueError(
                "out_channels has to be greater than the in_channels")

        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels -
                              in_channels, kernel_size=3, padding=1, stride=2)
        self.pool = nn.MaxPool2d(kernel_size=2)

        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.PReLU()

    def forward(self, input):
        x_conv = self.conv(input)
        x_pool = self.pool(input)

        x = torch.cat([x_conv, x_pool], dim=1)
        x = self.bn(x)

        return self.activation(x)


class RegularBottleneck(nn.Module):

    def __init__(self, in_channels, out_channels,
                 kernel_size=3, dilation=1, projection_ratio=4, dropout_p=0.1):
        super().__init__()

        mid_channels = in_channels // projection_ratio

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.PReLU(),
        )

        if kernel_size == 3:
            self.conv2 = nn.Sequential(
                nn.Conv2d(mid_channels, mid_channels, kernel_size=3,
                          padding=dilation, dilation=dilation, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.PReLU(),
            )
        elif kernel_size == 5:
            self.conv2 = nn.Sequential(
                nn.Conv2d(mid_channels, mid_channels, kernel_size=(
                    1, 5), padding=(0, 2), bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.Conv2d(mid_channels, mid_channels, kernel_size=(
                    5, 1), padding=(2, 0), bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.PReLU(),
            )
        else:
            raise ValueError("kernel_size must be either 3 or 5")

        self.conv3 = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

        self.dropout = nn.Dropout2d(p=dropout_p)

        self.activation = nn.PReLU()

    def forward(self, input):

        x = self.conv1(input)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.dropout(x)

        if input.shape != x.shape:
            b, _, h, w = input.shape
            c = x.size(1) - input.size(1)
            pad = torch.zeros((b, c, h, w), dtype=x.dtype, device=x.device)
            input = torch.cat([input, pad], dim=1)

        return self.activation(input + x)


class DownsamplingBottleneck(nn.Module):

    def __init__(self, in_channels, out_channels,
                 projection_ratio=4, dropout_p=0.1):
        super().__init__()

        mid_channels = in_channels // projection_ratio

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels,
                      kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.PReLU(),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3,
                      padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.PReLU(),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

        self.dropout = nn.Dropout2d(p=dropout_p)

        self.activation = nn.PReLU()

        self.pool = nn.MaxPool2d(kernel_size=2, return_indices=True)

    def forward(self, input):
        x = self.conv1(input)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.dropout(x)

        s, indices = self.pool(input)

        if s.shape != x.shape:
            b, _, h, w = s.shape
            c = x.size(1) - s.size(1)
            pad = torch.zeros((b, c, h, w), dtype=x.dtype, device=x.device)
            s = torch.cat([s, pad], dim=1)

        return self.activation(s + x), indices


class UpsamplingBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, projection_ratio=4, dropout_p=0.1):
        super().__init__()

        reduced_depth = in_channels // projection_ratio

        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels, reduced_depth,
                               kernel_size=1, bias=False),
            nn.BatchNorm2d(reduced_depth),
            nn.PReLU(),
        )

        self.conv2 = nn.Sequential(
            nn.ConvTranspose2d(reduced_depth, reduced_depth, kernel_size=3,
                               stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(reduced_depth),
            nn.PReLU(),
        )

        self.conv3 = nn.Sequential(
            nn.ConvTranspose2d(reduced_depth, out_channels,
                               kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Dropout2d(p=dropout_p)
        )

        self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.pad = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        self.activation = nn.PReLU()

    def forward(self, input, indices):
        s = self.conv1(input)
        s = self.conv2(s)
        s = self.conv3(s)

        x = self.pad(input)
        x = self.unpool(x, indices, output_size=s.size())

        x = x + s
        x = self.activation(x)
        return x
