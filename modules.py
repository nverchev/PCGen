import torch
import torch.nn as nn

act = nn.LeakyReLU


class Transpose(nn.Module):

    def forward(self, x):
        return x.transpose(2, 1)


class View(nn.Module):

    def __init__(self, *shape):
        super().__init__()
        self.shape = shape

    def forward(self, input):
        return input.view(self.shape)


class MaxChannel(nn.Module):

    def forward(self, x):
        return torch.max(x, 1)[0]


# function pretending to be a class
def get_points_batch_norm(dim):
    return nn.Sequential(Transpose(), nn.BatchNorm1d(dim), Transpose())


class PointsConvBlock(nn.Module):
    # Dense + Batch + Relu
    def __init__(self, in_dim, out_dim, act=act()):
        super().__init__()
        self.dense = nn.Linear(in_dim, out_dim)
        self.bn = get_points_batch_norm(out_dim)
        self.act = act
        self.in_dim = in_dim
        self.out_dim = out_dim

    def forward(self, x):
        x = self.bn(self.dense(x))
        return x if self.act is None else self.act(x)


# Deals with features, no need of transposing during batch normalization
class LinearBlock(PointsConvBlock):
    # Dense + Batch + Relu
    def __init__(self, in_dim, out_dim, act=act()):
        super().__init__(in_dim, out_dim, act=act)
        self.bn = nn.BatchNorm1d(out_dim)


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


def get_conv2d(in_dim, out_dim):
    return nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size=1, bias=False), nn.BatchNorm2d(out_dim), act())


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

