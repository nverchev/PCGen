import torch
import torch.nn as nn
from src.layer import PointsConvLayer, LinearLayer, EdgeConvLayer
from src.neighbour_op import get_graph_features, graph_max_pooling, get_local_covariance

IN_CHAN = 3


class CWEncoder(nn.Module):

    def __init__(self, cw_dim, z_dim, dim_embedding):
        super().__init__()
        self.cw_dim = cw_dim
        self.project_dim = 16 * dim_embedding
        self.h_dim = [cw_dim * self.project_dim // dim_embedding]
        self.conv = PointsConvLayer(dim_embedding,  self.project_dim)
        self.encode = nn.Sequential(LinearLayer(self.h_dim[0], self.cw_dim),
                                    nn.Dropout(0.3),)

    def forward(self, x):
        x = self.conv(x).view(-1, self.h_dim[0])
        x = self.encode(x)
        return x


class CWEncoder(nn.Module):

    def __init__(self, cw_dim, z_dim, dim_embedding):
        super().__init__()
        self.cw_dim = cw_dim
        self.project_dim = 16 * dim_embedding
        self.h_dim = [cw_dim * self.project_dim // dim_embedding]
        self.conv = PointsConvLayer(dim_embedding,  self.project_dim)
        self.encode = nn.Sequential(LinearLayer(self.h_dim[0], self.h_dim[0]),
                                    nn.Dropout(0.3),
                                    LinearLayer(self.h_dim[0], 2 * z_dim, batch_norm=False, act=None))

    def forward(self, x):
        x = self.conv(x).view(-1, self.h_dim[0])
        x = self.encode(x)
        return x
#
# class CWEncoder(nn.Module):
#
#     def __init__(self, cw_dim, z_dim, dim_embedding):
#         super().__init__()
#         self.cw_dim = cw_dim
#         self.dim_embedding = dim_embedding
#         self.project_dim = 32
#         self.h_dim = [cw_dim * self.project_dim // dim_embedding]
#         self.conv = PointsConvLayer(dim_embedding, self.project_dim)
#         self.att1 = nn.MultiheadAttention(self.project_dim, num_heads=4, batch_first=True, dropout=0.3)
#         self.att2 = nn.MultiheadAttention(self.project_dim, num_heads=4, batch_first=True, dropout=0.3)
#
#         self.encode = nn.Sequential(nn.ReLU(),
#                                     nn.BatchNorm1d(self.project_dim),
#                                     PointsConvLayer(self.project_dim, 2, batch_norm=False, act=None),)
#                                     #LinearLayer(self.h_dim[0], 2 * z_dim, batch_norm=False, act=None))
#
#     def forward(self, x):
#         b = x.shape[0]
#         x = self.conv(x).transpose(2, 1)
#         x = self.att1(x, x, x, need_weights=False)[0] + x
#         x = self.att2(x, x, x, need_weights=False)[0].transpose(2, 1)
#         x = self.encode(x).reshape(b, 2 * self.cw_dim // self.dim_embedding)
#         return x

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
