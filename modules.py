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

    def forward(self, x):
        return torch.max(x, 1)[0]


class PointBatch1D(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.batchnorm = nn.BatchNorm1d(out_dim)

    def forward(self, x):
        x = x.transpose(-1, -2)
        x = self.batchnorm(x)
        return x.transpose(-1, -2)


class PointBatch2D(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.batchnorm = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        x = x.transpose(-1, -3)
        x = self.batchnorm(x)
        return x.transpose(-1, -3)


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
        self.bn = PointBatch1D(out_dim)


# Input (Batch, Samples, Points, Features)
class PointsConvBlock4(PointsConvBlock):

    def forward(self, x, n_samples=1, m=None):
        if m is not None:
            x = x.view(-1, m, self.in_dim)
        else:
            x = x.squeeze()
        x = super().forward(x)
        if m is not None:
            x = x.view(-1, n_samples, m, self.out_dim)
        return x


# Input (Batch, Channels, Edge_start, Edge_end)
class EdgeConvBlock(LinearBlock):

    def __init__(self, in_dim, out_dim, act=act):
        super().__init__(in_dim, out_dim, act)
        self.bn = PointBatch2D(out_dim)


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
