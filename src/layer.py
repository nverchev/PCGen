import functools
import torch
import torch.nn as nn
from torch.autograd import Function

negative_slope = 0.2
Act = functools.partial(nn.LeakyReLU, negative_slope=0.2)


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
class LinearLayer(nn.Module):

    def __init__(self, in_dim, out_dim, Act=Act, batch_norm=True, groups=1):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.groups = groups
        self.batch_norm = batch_norm
        self.bias = True
        self.bn = self.get_bn_layer() if batch_norm else None
        self.bias = False if batch_norm else True
        self.dense = self.get_dense_layer()
        self.act = None if Act is None else Act(inplace=True)
        self.init(self.act)

    def init(self, act):
        if act is None:
            # nn.init.xavier_uniform_(self.dense.weight, gain=1)
            pass  # default works better than theoretically approved one
        elif act._get_name() == 'ReLU':
            # nn.init.kaiming_uniform_(self.dense.weight, nonlinearity='relu')
            pass  # default works better than theoretically approved one
        elif act._get_name() == 'LeakyReLU':
            nn.init.kaiming_uniform_(self.dense.weight, a=negative_slope)
            pass
        elif act._get_name() == 'Hardtanh':
            # nn.init.xavier_normal_(self.dense.weight, gain=nn.init.calculate_gain('tanh'))
            pass  # default works better than theoretically approved one

    def get_dense_layer(self):
        if self.groups > 1:
            raise NotImplementedError('nn.Linear has not option for groups')
        return nn.Linear(self.in_dim, self.out_dim, bias=self.bias)

    def get_bn_layer(self):
        return nn.BatchNorm1d(self.out_dim)

    def forward(self, x):
        x = self.bn(self.dense(x)) if self.batch_norm else self.dense(x)
        return x if self.act is None else self.act(x)


# Input (Batch, Points, Features)
class PointsConvLayer(LinearLayer):

    def get_dense_layer(self):
        return nn.Conv1d(self.in_dim, self.out_dim, kernel_size=1, bias=self.bias, groups=self.groups)


class EdgeConvLayer(LinearLayer):

    def get_dense_layer(self):
        return nn.Conv2d(self.in_dim, self.out_dim, kernel_size=1, bias=self.bias, groups=self.groups)

    def get_bn_layer(self):
        return nn.BatchNorm2d(self.out_dim)


class TransferGrad(Function):

    @staticmethod
    # transfer the grad from output to input during backprop
    def forward(ctx, input, output):
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None