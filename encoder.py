import numpy as nn
import torch
from modules import DBR, DbR, STN, MaxChannel, get_points_batch_norm
from pointnet_modules import PCT, SPCT
# input feature dimension
IN_CHAN = 3
N_POINTS = 2048
Z_DIM = 128


def knn(x, k):
    batch_size = x.size(0)
    num_points = x.size(2)

    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)

    if idx.get_device() == -1:
        idx_base = torch.arange(0, batch_size).view(-1, 1, 1) * num_points
    else:
        idx_base = torch.arange(0, batch_size, device=idx.get_device()).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)

    return idx


def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)  # (batch_size, num_dims, num_points)
    if idx is None:
        idx = knn(x, k=k)  # (batch_size, num_points, k)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()  # (batch_size, num_points, num_dims)
    feature = x.view(batch_size * num_points, -1)[idx, :]  # (batch_size*n, num_dims) -> (batch_size*n*k, num_dims)
    feature = feature.view(batch_size, num_points, k, num_dims)  # (batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)  # (batch_size, num_points, k, num_dims)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1,
                                                         2).contiguous()  # (batch_size, num_points, k, 2*num_dims) -> (batch_size, 2*num_dims, num_points, k)

    return feature  # (batch_size, 2*num_dims, num_points, k)


class DGCNN_Cls_Encoder(nn.Module):
    def __init__(self, feat_dim=1024, c_dim=128, k=20):
        super(DGCNN_Cls_Encoder, self).__init__()

        self.k = k
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(feat_dim)
        self.bn6 = nn.BatchNorm1d(512)
        self.bn7 = nn.BatchNorm1d(2*c_dim)

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
        self.conv5 = nn.Sequential(nn.Conv1d(512, feat_dim, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.conv6 = nn.Sequential(nn.Conv1d(feat_dim + 512, 512, kernel_size=1, bias=False),
                                   self.bn6,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv7 = nn.Sequential(nn.Conv1d(512, 2 * c_dim, kernel_size=1, bias=False),
                                   self.bn7,
                                   nn.LeakyReLU(negative_slope=0.2))

    def forward(self, x):
        x = x.transpose(2, 1).contiguous()

        batch_size, _, num_points = x.size()
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

        x = self.conv5(x)  # (batch_size, 512, num_points) -> (batch_size, feat_dims, num_points)
        x = x.max(dim=-1, keepdim=True)[0]  # (batch_size, feat_dims, num_points) -> (batch_size, feat_dims)

        x = x.repeat(1, 1, num_points)
        x = torch.cat((x, x1, x2, x3, x4), dim=1)

        x = self.conv6(x)
        x = self.conv7(x)

        return x.permute(0, 2, 1).contiguous().max(dim=1)[0]





def get_mlp_encoder():
    h_chan = [64, 64, 64, 128, 128, 512]
    modules = [DBR(IN_CHAN, h_chan[0])]
    for i in range(len(h_chan) - 2):
        modules.append(DBR(h_chan[i], h_chan[i + 1]))
    modules.append(MaxChannel())
    modules.append(DbR(h_chan[-2], h_chan[-1]))
    modules.append(nn.Linear(h_chan[-1], 2 * Z_DIM))
    return nn.Sequential(*modules)


class PointNetEncoder(nn.Module):

    def __init__(self):
        super().__init__()
        h_chan = [64, 128, 128, 256]
        self.stn = STN()
        self.stnk = STN(h_chan[0])
        self.dbr1 = DBR(IN_CHAN, h_chan[0])
        modules = [DBR(h_chan[0], h_chan[1]),
                   nn.Linear(h_chan[1], h_chan[2]),
                   get_points_batch_norm(h_chan[2]),
                   MaxChannel(),
                   DbR(self.h_chan[2], self.h_chan[3]),
                   nn.Linear(self.h_chan[3], 2 * Z_DIM)]
        self.encode = nn.Sequential(*modules)

    def forward(self, x):
        trans = self.stn(x)
        x = torch.matmul(x, trans)
        x = self.dbr1(x)
        trans = self.stnk(x)
        x = torch.matmul(x, trans)
        x = self.encode(x)
        return [x, trans]

    class DGCNN(nn.Module):
        pass


def get_encoder(encoder_name):
    dict_encoder = {
        "MLP": get_mlp_encoder(),
        "PointNet": PointNetEncoder,
        "DGCNN": DGCNN,
        "PCT": PCT
    }
    return dict_encoder[encoder_name]
