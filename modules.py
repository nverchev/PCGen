import torch
import torch.nn as nn


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


class DBR(nn.Module):
    # Dense + Batch + Relu
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.dense = nn.Linear(in_dim, out_dim)
        self.bn = get_points_batch_norm(out_dim)
        self.relu = nn.ReLU()
        self.in_dim = in_dim
        self.out_dim = out_dim

    def forward(self, x):
        return self.relu(self.bn(self.dense(x)))


class DBR4(DBR):

    def forward(self, x, n_samles, m):
        x = x.view(-1, m, self.in_dim)
        x = super().forward(x)
        x = x.view(-1, n_samles, m, self.out_dim)
        return x


def get_conv2d(in_dim, out_dim):
    return nn.Sequential(nn.Conv2d(in_dim , out_dim, kernel_size=1, bias=False), nn.BatchNorm2d(out_dim), nn.ReLU())

# Deals with features, no need of transposing
class DbR(nn.Module):
    # Dense + Batch + Relu
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.dense = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.dense(x)))


class STN(nn.Module):
    def __init__(self, channels=3):
        super().__init__()
        self.channels = channels
        self.net = nn.Sequential(DBR(channels, 64),
                                 DBR(64, 128),
                                 DBR(128, 1024),
                                 MaxChannel(),
                                 DbR(1024, 512),
                                 DbR(512, 256),
                                 nn.Linear(256, channels ** 2))

        self.register_buffer('eye', torch.eye(channels))  # changes device automatically

    def forward(self, x):
        x = self.net(x).view(-1, self.channels, self.channels)
        x += self.eye
        return x
