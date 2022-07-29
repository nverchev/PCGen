import torch
import torch.nn as nn
from modules import PointsConvBlock, LinearBlock, STN, MaxChannel, EdgeConvBlock
from utils import get_graph_features

# from pointnet_modules import PCT, SPCT
# input feature dimension
IN_CHAN = 3
N_POINTS = 2048
Z_DIM = 512


class DGCNN_sim(nn.Module):
    def __init__(self, k=20):
        super().__init__()
        self.k = k
        h_dim = [64, 64, 64, 128, 128, 512]
        self.conv = EdgeConvBlock(2 * IN_CHAN, h_dim[0])
        modules = []
        for i in range(1, len(h_dim) - 2):
            modules.append(PointsConvBlock(h_dim[i], h_dim[i + 1]))
        modules.append(MaxChannel())
        modules.append(LinearBlock(h_dim[-2], h_dim[-1]))
        modules.append(nn.Linear(h_dim[-1], 2 * Z_DIM))
        self.encode = nn.Sequential(*modules)

    def forward(self, x):
        x = x.transpose(2, 1).contiguous()
        x = get_graph_features(x, k=self.k)
        x = self.conv(x)
        x = x.max(dim=-1, keepdim=False)[0]
        x = x.transpose(2, 1).contiguous()
        return self.encode(x)


class DGCNN(nn.Module):
    def __init__(self, k=40):
        super().__init__()
        self.k = k
        h_dim = [64, 64, 128, 256]
        edge_conv_list = [EdgeConvBlock(2 * IN_CHAN, h_dim[0])]
        for i in range(len(h_dim) - 3):
            edge_conv_list.append(EdgeConvBlock(2 * h_dim[i], h_dim[i + 1]))
        self.edge_convs = nn.Sequential(*edge_conv_list)
        self.final_conv = nn.Linear(sum(h_dim), 2 * Z_DIM)

    def forward(self, x):
        x = x.transpose(2, 1).contiguous()
        xs = []
        for conv in self.edge_convs:
            x = get_graph_features(x, k=self.k)
            x = conv(x)
            x = x.max(dim=-1, keepdim=False)[0]
            xs.append(x)
        x = torch.cat(xs, dim=1).transpose(2, 1).contiguous()
        x = self.final_conv(x)
        x = x.max(dim=1, keepdim=False)[0]

        return x


def get_mlp_encoder():
    h_dim = [64, 64, 128, 256, 128, 512]
    modules = [PointsConvBlock(IN_CHAN, h_dim[0])]
    for i in range(len(h_dim) - 2):
        modules.append(PointsConvBlock(h_dim[i], h_dim[i + 1]))
    modules.append(MaxChannel())
    modules.append(LinearBlock(h_dim[-2], h_dim[-1]))
    modules.append(nn.Linear(h_dim[-1], 2 * Z_DIM))
    return nn.Sequential(*modules)


def get_encoder(encoder_name):
    dict_encoder = {
        "MLP": get_mlp_encoder,
        "DGCNN_sim": DGCNN_sim,
        "DGCNN": DGCNN,
        # "PCT": PCT
    }
    return dict_encoder[encoder_name]
