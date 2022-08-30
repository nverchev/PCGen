import torch
import torch.nn as nn

act = nn.LeakyReLU(negative_slope=0.2)


class View(nn.Module):

    def __init__(self, *shape):
        super().__init__()
        self.shape = shape

    def forward(self, inputs):
        return inputs.view(self.shape)


class MaxChannel(nn.Module):

    def forward(self, x, axis=-1):
        return torch.max(x, axis)[0]


# Input (Batch, Features)
class LinearBlock(nn.Module):

    # Dense + Batch + Act
    def __init__(self, in_dim, out_dim, act=act):
        super().__init__()
        self.dense = nn.Linear(in_dim, out_dim, bias=False)
        self.bn = nn.BatchNorm1d(out_dim)
        self.act = act
        self.in_dim = in_dim
        self.out_dim = out_dim

    def forward(self, x):
        x = self.bn(self.dense(x))
        return x if self.act is None else self.act(x)


# Input (Batch, Points, Features)
class PointsConvBlock(LinearBlock):
    # Dense + Batch + Relu
    def __init__(self, in_dim, out_dim, act=act):
        super().__init__(in_dim, out_dim, act)
        self.dense = nn.Conv1d(in_dim, out_dim, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(out_dim)


class EdgeConvBlock(LinearBlock):

    def __init__(self, in_dim, out_dim, act=act):
        super().__init__(in_dim, out_dim, act)
        self.dense = nn.Conv2d(in_dim, out_dim, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_dim)


class STN(nn.Module):

    def __init__(self, channels=3):
        super().__init__()
        self.channels = channels
        self.net = nn.Sequential(PointsConvBlock(channels, 64),
                                 PointsConvBlock(64, 128),
                                 PointsConvBlock(128, 1024),
                                 MaxChannel(),
                                 LinearBlock(1024, 512),
                                 LinearBlock(512, 256),
                                 nn.Linear(256, channels ** 2))

        self.register_buffer('eye', torch.eye(channels))  # changes device automatically

    def forward(self, x):
        x = self.net(x).view(-1, self.channels, self.channels)
        x += self.eye
        return x
