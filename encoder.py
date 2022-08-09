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
    def __init__(self, k=40, task='reconstruct'):
        super().__init__()
        self.k = k
        h_dim = [64, 64, 128, 256]
        edge_conv_list = [EdgeConvBlock(2 * IN_CHAN, h_dim[0])]
        for i in range(len(h_dim) - 1):
            edge_conv_list.append(EdgeConvBlock(2 * h_dim[i], h_dim[i + 1]))
        self.edge_convs = nn.Sequential(*edge_conv_list)
        self.final_conv = nn.Linear(sum(h_dim), 2 * Z_DIM)
        self.task = task

    def forward(self, x):
        xs = []
        for conv in self.edge_convs:
            x = get_graph_features(x, k=self.k) #[batch, num_points, k, features]
            x = conv(x)
            x = x.max(dim=2, keepdim=False)[0] #[batch, num_points, features]
            xs.append(x)
        x = torch.cat(xs, dim=2).contiguous()
        x = self.final_conv(x)
        x_max = x.max(dim=1, keepdim=False)[0]
        if self.task == 'reconstruct':
            x = x_max
        if self.task == 'classify':
            x_avg = x.mean(dim=1)
            x = torch.cat([x_max, x_avg], dim=1)
        return x




def get_encoder(encoder_name):
    dict_encoder = {
        "DGCNN_sim": DGCNN_sim,
        "DGCNN": DGCNN,
        # "PCT": PCT
    }
    return dict_encoder[encoder_name]
