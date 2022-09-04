import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from src.modules import PointsConvBlock, LinearBlock, MaxChannel, EdgeConvBlock
from src.utils import get_graph_features, graph_max_pooling, get_local_covariance


class DGCNN_Vanilla(nn.Module):
    def __init__(self, in_chan=3, z_dim=512, k=20, log_var=True):
        super().__init__()
        self.k = k
        self.h_dim = [64, 64, 128, 256]
        self.edge_conv = EdgeConvBlock(2 * in_chan, self.h_dim[0])
        modules = []
        for i in range(3):
            modules.append(PointsConvBlock(self.h_dim[i], self.h_dim[i + 1]))
        self.points_convs = nn.Sequential(*modules)
        self.final_conv = nn.Conv1d(sum(self.h_dim), 2 * z_dim if log_var else z_dim, kernel_size=1)

    def forward(self, x):
        x = x.transpose(2, 1)
        indices = None
        if x.size(1) > 3:
            x, indices = x[:, :3, :],  x[:, 3:, :].long()
        x = get_graph_features(x, k=self.k, indices=indices)
        x = self.edge_conv(x)
        x = x.max(dim=3, keepdim=False)[0]
        xs = [x]
        for conv in self.points_convs:
            x = graph_max_pooling(x)
            x = conv(x)
            xs.append(x)
        x = torch.cat(xs, dim=1).contiguous()
        x = self.final_conv(x)
        x_max = x.max(dim=2, keepdim=False)[0]
        return x_max


class DGCNN(nn.Module):
    def __init__(self, in_chan=3, z_dim=512, k=20, log_var=True):
        super().__init__()
        self.k = k
        self.h_dim = [64, 64, 128, 256]
        edge_conv_list = [EdgeConvBlock(2 * in_chan, self.h_dim[0])]
        for i in range(len(self.h_dim) - 1):
            edge_conv_list.append(EdgeConvBlock(2 * self.h_dim[i], self.h_dim[i + 1]))
        self.edge_convs = nn.Sequential(*edge_conv_list)
        self.final_conv = nn.Conv1d(sum(self.h_dim), 2 * z_dim if log_var else z_dim, kernel_size=1)

    def forward(self, x):
        xs = []
        x = x.transpose(2, 1)
        indices = None
        if x.size(1) > 3:
            x, indices = x[:, :3, :],  x[:, 3:, :].long()
        for conv in self.edge_convs:
            x = get_graph_features(x, k=self.k, indices=indices)  # [batch, features, num_points, k]
            indices = None  # finds new neighbours dynamically after first iteration
            x = conv(x)
            x = x.max(dim=3, keepdim=False)[0]  # [batch, features, num_points]
            xs.append(x)
        x = torch.cat(xs, dim=1).contiguous()
        x = self.final_conv(x)
        x_max = x.max(dim=2, keepdim=False)[0]
        x = x_max
        return x


class FoldingNet(nn.Module):
    def __init__(self, in_chan=3, z_dim=512, k=16, log_var=True):
        super().__init__()
        self.k = k
        self.h_dim = [64, 64, 64, 128, 1024, 1024]
        modules = [PointsConvBlock(in_chan + in_chan ** 2, self.h_dim[0], act=nn.ReLU())]
        for i in range(2):
            modules.append(PointsConvBlock(self.h_dim[i], self.h_dim[i + 1], act=nn.ReLU()))
        self.point_mlp = nn.Sequential(*modules)
        self.conv1 = PointsConvBlock(self.h_dim[2], self.h_dim[3], act=nn.ReLU())
        self.conv2 = PointsConvBlock(self.h_dim[3], self.h_dim[4], act=nn.ReLU())
        self.features_mlp = nn.Sequential(LinearBlock(self.h_dim[4], self.h_dim[5], act=nn.ReLU()),
                                          nn.Linear(self.h_dim[5], 2 * z_dim if log_var else z_dim,))

    def forward(self, x):
        x = x.transpose(2, 1)  # (batch_size, 3, num_points)
        indices = None
        if x.size(1) > 3:
            x, indices = x[:, :3, :],  x[:, 3:, :].long()
        x = get_local_covariance(x, self.k, indices)
        x = self.point_mlp(x)
        x = graph_max_pooling(x)
        x = self.conv1(x)
        x = graph_max_pooling(x)
        x = self.conv2(x)
        x = x.mean(-1)
        x = self.features_mlp(x)
        return x


def get_encoder(encoder_name):
    dict_encoder = {
        'DGCNN_Vanilla': DGCNN_Vanilla,
        'DGCNN': DGCNN,
        'FoldingNet': FoldingNet,
    }
    return dict_encoder[encoder_name]
