import torch
import torch.nn as nn

negative_slope = 0.2
act = nn.LeakyReLU(negative_slope=0.2, inplace=True)


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

    def __init__(self, in_dim, out_dim, act=act, batch_norm=True, groups=1):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.groups = groups
        self.batch_norm = batch_norm
        self.bias = True
        if batch_norm:
            self.bn = self.get_bn_layer()
            self.bias = False
        self.dense = self.get_dense_layer()
        self.act = act
        self.init(act)

    def init(self, act):
        if act is None:
            pass
            # nn.init.xavier_uniform_(self.dense.weight, gain=1)
        elif act._get_name() == 'ReLU':
            # nn.init.kaiming_uniform_(self.dense.weight, nonlinearity='relu')
            pass  # default works better than theoretically approved one
        elif act._get_name() == 'LeakyReLU':
            # nn.init.kaiming_uniform_(self.dense.weight, a=negative_slope)
            pass  # default works better than theoretically approved one
        elif act._get_name() == 'Hardtanh':
            # nn.init.xavier_normal_(self.dense.weight, gain=nn.init.calculate_gain('tanh'))
            pass  # default works better than theoretically approved one

    def get_dense_layer(self):
        return nn.Linear(self.in_dim, self.out_dim, bias=self.bias)

    def get_bn_layer(self):
        return nn.BatchNorm1d(self.out_dim)

    def forward(self, x):
        x = self.dense(x) if self.batch_norm is False else self.bn(self.dense(x))
        return x if self.act is None else self.act(x)


# Input (Batch, Points, Features)
class PointsConvBlock(LinearBlock):

    def get_dense_layer(self):
        return nn.Conv1d(self.in_dim, self.out_dim, kernel_size=1, bias=self.bias, groups=self.groups)


class EdgeConvBlock(LinearBlock):

    def get_dense_layer(self):
        return nn.Conv2d(self.in_dim, self.out_dim, kernel_size=1, bias=self.bias, groups=self.groups)

    def get_bn_layer(self):
        return nn.BatchNorm2d(self.out_dim)


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
