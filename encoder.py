import torch
import torch.nn as nn
from modules import PointsConvBlock, LinearBlock, STN, MaxChannel, EdgeConvBlock
from utils import get_graph_features, get_graph_feature

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
        for i in range(len(h_dim) - 1):
            edge_conv_list.append(EdgeConvBlock(2 * h_dim[i], h_dim[i + 1]))
        self.edge_convs = nn.Sequential(*edge_conv_list)
        self.final_conv = nn.Linear(sum(h_dim), 2 * Z_DIM)

    def forward(self, x):
        x = x.transpose(2, 1).contiguous()
        xs = []
        for conv in self.edge_convs:
            x = get_graph_features(x, k=self.k)
            torch.cuda.synchronize()
            print("Memory:", torch.cuda.memory_allocated())
            x = conv(x)
            x = x.max(dim=-1, keepdim=False)[0]
            xs.append(x)
        x = torch.cat(xs, dim=1).transpose(2, 1).contiguous()
        x = self.final_conv(x)
        x = x.max(dim=1, keepdim=False)[0]

        return x


class DGCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.k = 40

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(1028)

        self.conv1 = nn.Sequential(nn.Conv2d(3 * 2, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, args.feat_dims, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))

    def forward(self, x):
        x = x.transpose(2, 1)
        x = get_graph_feature(x, k=self.k)  # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        x = self.conv1(x)  # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x1, k=self.k)  # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv2(x)  # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x2, k=self.k)  # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv3(x)  # (batch_size, 64*2, num_points, k) -> (batch_size, 128, num_points, k)
        x3 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 128, num_points, k) -> (batch_size, 128, num_points)

        x = get_graph_feature(x3, k=self.k)  # (batch_size, 128, num_points) -> (batch_size, 128*2, num_points, k)
        x = self.conv4(x)  # (batch_size, 128*2, num_points, k) -> (batch_size, 256, num_points, k)
        x4 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 256, num_points, k) -> (batch_size, 256, num_points)

        x = torch.cat((x1, x2, x3, x4), dim=1)  # (batch_size, 512, num_points)

        x0 = self.conv5(x)  # (batch_size, 512, num_points) -> (batch_size, feat_dims, num_points)
        x = x0.max(dim=-1, keepdim=False)[0]  # (batch_size, feat_dims, num_points) -> (batch_size, feat_dims)
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
