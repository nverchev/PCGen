import torch
import torch.nn as nn
import torch.nn.init as init
from modules import PointsConvBlock, LinearBlock, MaxChannel, EdgeConvBlock
from utils import get_graph_features

# input feature dimension
IN_CHAN = 3
N_POINTS = 2048
Z_DIM = 512


class DGCNN_sim(nn.Module):
    def __init__(self, k=20):
        super().__init__()
        self.k = k
        self.h_dim = [64, 64, 64, 128, 128, 512]
        self.conv = EdgeConvBlock(2 * IN_CHAN, self.h_dim[0])
        modules = []
        for i in range(1, len(self.h_dim) - 2):
            modules.append(PointsConvBlock(self.h_dim[i], self.h_dim[i + 1]))
        modules.append(MaxChannel())
        modules.append(LinearBlock(self.h_dim[-2], self.h_dim[-1]))
        modules.append(nn.Linear(self.h_dim[-1], 2 * Z_DIM))
        self.encode = nn.Sequential(*modules)

    def forward(self, x):
        x = x.transpose(2, 1)
        x = get_graph_features(x, k=self.k)
        x = self.conv(x)
        x = x.max(dim=3, keepdim=False)[0]
        return self.encode(x)


class DGCNN(nn.Module):
    def __init__(self, k=40, task='reconstruct'):
        super().__init__()
        self.k = k
        self.h_dim = [64, 64, 128, 256]
        edge_conv_list = [EdgeConvBlock(2 * IN_CHAN, self.h_dim[0])]
        for i in range(len(self.h_dim) - 1):
            edge_conv_list.append(EdgeConvBlock(2 * self.h_dim[i], self.h_dim[i + 1]))
        self.edge_convs = nn.Sequential(*edge_conv_list)
        self.final_conv = nn.Conv1d(sum(self.h_dim), 2 * Z_DIM, kernel_size=1)
        self.task = task

    def forward(self, x):
        xs = []
        x = x.transpose(2, 1)
        for conv in self.edge_convs:
            x = get_graph_features(x, k=self.k) #[batch, features, num_points, k]
            x = conv(x)
            x = x.max(dim=3, keepdim=False)[0] #[batch, features, num_points]
            xs.append(x)
        x = torch.cat(xs, dim=1).contiguous()
        x = self.final_conv(x)
        x_max = x.max(dim=2, keepdim=False)[0]
        if self.task == 'reconstruct':
            x = x_max
        if self.task == 'classify':
            x_avg = x.mean(dim=2)
            x = torch.cat([x_max, x_avg], dim=1).contiguous()
        return x


class Point_Transform_Net(nn.Module):
    def __init__(self):
        super(Point_Transform_Net, self).__init__()
        h_dim = [64, 128, 1024, 512, 256]
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.conv1 = EdgeConvBlock(2 * IN_CHAN, h_dim[0])
        self.conv2 = EdgeConvBlock(h_dim[0], h_dim[1])
        self.conv3 = EdgeConvBlock(h_dim[1], h_dim[2])
        self.linear1 = LinearBlock(h_dim[2], h_dim[3])
        self.linear2 = LinearBlock(h_dim[3], h_dim[4])
        self.transform = nn.Linear(h_dim[4], IN_CHAN ** 2)
        init.constant_(self.transform.weight, 0)
        init.eye_(self.transform.bias.view(3, 3))

    def forward(self, x):
        x = self.conv1(x)  # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv2(x)  # (batch_size, 64, num_points, k) -> (batch_size, 128, num_points, k)
        x = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 128, num_points, k) -> (batch_size, 128, num_points)
        x = self.conv3(x)  # (batch_size, 128, num_points) -> (batch_size, 1024, num_points)
        x = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 1024, num_points) -> (batch_size, 1024)
        x = self.linear1(x)  # (batch_size, 1024) -> (batch_size, 512)
        x = self.linear2(x)  # (batch_size, 512) -> (batch_size, 256)
        x = self.transform(x)  # (batch_size, 256) -> (batch_size, 3*3)
        x = x.view(-1, IN_CHAN, IN_CHAN)  # (batch_size, 3*3) -> (batch_size, 3, 3)

        return x  # (batch_size, 3, 3)


class DGCNN_Seg(nn.Module):
    def __init__(self,  k=40, task='reconstruct'):
        super().__init__()
        self.k = k
        self.transform_net = Point_Transform_Net()
        h_dim = [64] * 5 + [2 * Z_DIM]
        self.conv1 = EdgeConvBlock(2 * IN_CHAN, h_dim[0])
        self.conv2 = EdgeConvBlock(h_dim[0], h_dim[1])
        self.conv3 = EdgeConvBlock(2 * h_dim[1], h_dim[2])
        self.conv4 = EdgeConvBlock(h_dim[2], h_dim[3])
        self.conv5 = EdgeConvBlock(2 * h_dim[3], h_dim[4])
        self.conv6 = EdgeConvBlock(2 * h_dim[4], h_dim[5])

    def forward(self, x):
        x = x.transpose(2, 1)
        x0 = get_graph_features(x, k=self.k)  # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        t = self.transform_net(x0)  # (batch_size, 3, 3)
        x = x.transpose(2, 1)  # (batch_size, 3, num_points) -> (batch_size, num_points, 3)
        x = torch.bmm(x, t)  # (batch_size, num_points, 3) * (batch_size, 3, 3) -> (batch_size, num_points, 3)
        x = x.transpose(2, 1)  # (batch_size, num_points, 3) -> (batch_size, 3, num_points)
        x = get_graph_features(x, k=self.k)  # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        x = self.conv1(x)  # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)
        x = get_graph_features(x1, k=self.k)  # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv3(x)  # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)
        x = get_graph_features(x2, k=self.k)  # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv5(x)  # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x3 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)
        x = torch.cat((x1, x2, x3), dim=1)  # (batch_size, 64*3, num_points)
        x = self.conv6(x)  # (batch_size, 64*3, num_points) -> (batch_size, emb_dims, num_points)
        x = x.max(dim=-1, keepdim=False)[0]  # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims)
        return x  # (batch_size,  emb_dims)


def get_encoder(encoder_name):
    dict_encoder = {
        "DGCNN_sim": DGCNN_sim,
        "DGCNN": DGCNN,
        "DGCNN_Seg": DGCNN_Seg,
    }
    return dict_encoder[encoder_name]
