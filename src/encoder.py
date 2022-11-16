import torch
import torch.nn as nn
from src.layer import PointsConvLayer, LinearLayer, EdgeConvLayer
from src.neighbour_op import get_graph_features, graph_max_pooling, get_local_covariance

IN_CHAN = 3


class CWEncoder(nn.Module):

    def __init__(self, cw_dim, z_dim):
        super().__init__()
        self.cw_dim = cw_dim
        self.conv = nn.Sequential(nn.Conv1d(1, 2, kernel_size=4, stride=4),
                                  nn.BatchNorm1d(2),
                                  nn.LeakyReLU(inplace=True))
        self.h_dim = [z_dim * 4, z_dim * 2]
        modules = [LinearLayer(cw_dim // 2, self.h_dim[0])]
        for in_dim, out_dim in zip(self.h_dim[:-1], self.h_dim[1:]):
            modules.append(LinearLayer(in_dim, out_dim))
        modules.append(LinearLayer(self.h_dim[-1], 2 * z_dim, act=None, batch_norm=False))
        self.encode = nn.Sequential(*modules)

    def forward(self, x):
        x = self.conv(x.unsqueeze(1)).view(-1, self.cw_dim // 2)
        return self.encode(x)


class LDGCNN(nn.Module):
    def __init__(self, cw_dim, k, **model_settings):
        super().__init__()
        self.k = k
        self.h_dim = [64, 64, 128, 256]
        self.edge_conv = EdgeConvLayer(2 * IN_CHAN, self.h_dim[0])
        modules = []
        for in_dim, out_dim in zip(self.h_dim[:3], self.h_dim[1:]):
            modules.append(PointsConvLayer(in_dim, out_dim))
        self.points_convs = nn.Sequential(*modules)
        self.final_conv = PointsConvLayer(sum(self.h_dim), cw_dim, act=None, batch_norm=False)

    def forward(self, x, indices):
        x = x.transpose(2, 1)
        indices, x = get_graph_features(x, k=self.k, indices=indices)
        x = self.edge_conv(x)
        x = x.max(dim=3, keepdim=False)[0]
        xs = [x]
        for conv in self.points_convs:
            x = graph_max_pooling(x, k=self.k, indices=indices)
            x = conv(x)
            xs.append(x)
        x = torch.cat(xs, dim=1).contiguous()
        x = self.final_conv(x)
        x_max = x.max(dim=2, keepdim=False)[0]
        return x_max


class DGCNN(nn.Module):
    def __init__(self, cw_dim, k, **model_settings):
        super().__init__()
        self.k = k
        self.h_dim = [64, 64, 128, 256]
        edge_conv_list = [EdgeConvLayer(2 * IN_CHAN, self.h_dim[0], batch_norm=True)]
        for in_dim, out_dim in zip(self.h_dim[:3], self.h_dim[1:]):
            edge_conv_list.append(EdgeConvLayer(2 * in_dim, out_dim, batch_norm=True))
        self.edge_convs = nn.Sequential(*edge_conv_list)
        self.final_conv = nn.Conv1d(sum(self.h_dim), cw_dim, kernel_size=1)

    def forward(self, x, indices):
        xs = []
        x = x.transpose(2, 1)
        for conv in self.edge_convs:
            indices, x = get_graph_features(x, k=self.k, indices=indices)  # [batch, features, num_points, k]
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
    def __init__(self, cw_dim, k, **model_settings):
        super().__init__()
        self.k = k
        self.h_dim = [64, 64, 64, 128, 1024, 1024]
        modules = [PointsConvLayer(IN_CHAN + IN_CHAN ** 2, self.h_dim[0], act=nn.ReLU(), batch_norm=False)]
        for in_dim, out_dim in zip(self.h_dim[:2], self.h_dim[1:3]):
            modules.append(PointsConvLayer(in_dim, out_dim, act=nn.ReLU(), batch_norm=False))
        self.point_mlp = nn.Sequential(*modules)
        self.conv1 = PointsConvLayer(self.h_dim[2], self.h_dim[3], act=nn.ReLU(), batch_norm=False)
        self.conv2 = PointsConvLayer(self.h_dim[3], self.h_dim[4], act=nn.ReLU(), batch_norm=False)
        self.features_mlp = nn.Sequential(LinearLayer(self.h_dim[4], self.h_dim[5], act=nn.ReLU(), batch_norm=False),
                                          nn.Linear(self.h_dim[5], cw_dim))

    def forward(self, x, indices):
        x = x.transpose(2, 1)
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
        'LDGCNN': LDGCNN,
        'DGCNN': DGCNN,
        'FoldingNet': FoldingNet,
    }
    return dict_encoder[encoder_name]
