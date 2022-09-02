import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from src.modules import PointsConvBlock, LinearBlock, MaxChannel, EdgeConvBlock
from src.utils import get_graph_features

class DGCNN_Vanilla(nn.Module):
    def __init__(self, chan_in=3, z_dim=512, k=20, vq=False):
        super().__init__()
        self.k = k
        self.h_dim = [64, 128, 128, 1024, 512, 512]
        self.conv = EdgeConvBlock(2 * chan_in, self.h_dim[0])
        modules = []
        for i in range(3):
            modules.append(PointsConvBlock(self.h_dim[i], self.h_dim[i + 1]))
        modules.append(MaxChannel())
        for i in range(3, len(self.h_dim) - 1):
            modules.append(LinearBlock(self.h_dim[i], self.h_dim[i + 1]))
        modules.append(nn.Linear(self.h_dim[-1], z_dim if vq else 2 * z_dim))
        self.encode = nn.Sequential(*modules)

    def forward(self, x):
        x = x.transpose(2, 1)
        x = get_graph_features(x, k=self.k)
        x = self.conv(x)
        x = x.max(dim=3, keepdim=False)[0]
        return self.encode(x)


class DGCNN(nn.Module):
    def __init__(self, chan_in=3, z_dim=512, k=20, vq=False):
        super().__init__()
        self.k = k
        self.h_dim = [64, 64, 128, 256]
        edge_conv_list = [EdgeConvBlock(2 * chan_in, self.h_dim[0])]
        for i in range(len(self.h_dim) - 1):
            edge_conv_list.append(EdgeConvBlock(2 * self.h_dim[i], self.h_dim[i + 1]))
        self.edge_convs = nn.Sequential(*edge_conv_list)
        self.final_conv = nn.Conv1d(sum(self.h_dim), z_dim if vq else 2 * z_dim, kernel_size=1)

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
        x = x_max
        return x


class Point_Transform_Net(nn.Module):
    def __init__(self):
        super(Point_Transform_Net, self).__init__()
        h_dim = [64, 128, 1024, 512, 256]
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.conv1 = EdgeConvBlock(2 * chan_in, h_dim[0])
        self.conv2 = EdgeConvBlock(h_dim[0], h_dim[1])
        self.conv3 = EdgeConvBlock(h_dim[1], h_dim[2])
        self.linear1 = LinearBlock(h_dim[2], h_dim[3])
        self.linear2 = LinearBlock(h_dim[3], h_dim[4])
        self.transform = nn.Linear(h_dim[4], chan_in ** 2)
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
        x = x.view(-1, chan_in, chan_in)  # (batch_size, 3*3) -> (batch_size, 3, 3)

        return x  # (batch_size, 3, 3)




class FoldingNet(nn.Module):
    def __init__(self, chan_in=3, z_dim=512, k=16, vq=False):
        super().__init__()
        self.k = 16
        self.h_dim = [12, 64, 64, 64]

        self.n = 2048   # input point cloud size
        modules = PointsConvBlock(chan_in, 12, act=nn.ReLU())
        for i in range(len(self.h_dim) - 1):
            modules.append(PointsConvBlock(self.h_dim[i], self.h_dim[i + 1], act=nn.ReLU()))
            self.mlp1 = nn.Sequential(
            nn.Conv1d(12, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 64, 1),
            nn.ReLU(),
        )
        self.linear1 = nn.Linear(64, 64)
        self.conv1 = nn.Conv1d(64, 128, 1)
        self.linear2 = nn.Linear(128, 128)
        self.conv2 = nn.Conv1d(128, 1024, 1)
        self.mlp2 = nn.Sequential(
            nn.Conv1d(1024, 1024, 1),
            nn.ReLU(),
            nn.Conv1d(1024, 1024, 1),
        )

    def local_maxpool(self, x, idx):
        batch_size = x.size(0)
        num_points = x.size(2)
        x = x.view(batch_size, -1, num_points)

        _, num_dims, _ = x.size()

        x = x.transpose(2, 1).contiguous()  # (batch_size, num_points, num_dims)
        x = x.view(batch_size * num_points, -1)[idx, :]  # (batch_size*n, num_dims) -> (batch_size*n*k, num_dims)
        x = x.view(batch_size, num_points, -1, num_dims)  # (batch_size, num_points, k, num_dims)
        x, _ = torch.max(x, dim=2)  # (batch_size, num_points, num_dims)
        return x

    def local_cov(pts, idx):
        batch_size = pts.size(0)
        num_points = pts.size(2)
        pts = pts.view(batch_size, -1, num_points)  # (batch_size, 3, num_points)

        _, num_dims, _ = pts.size()

        x = pts.transpose(2, 1).contiguous()  # (batch_size, num_points, 3)
        x = x.view(batch_size * num_points, -1)[idx, :]  # (batch_size*num_points*2, 3)
        x = x.view(batch_size, num_points, -1, num_dims)  # (batch_size, num_points, k, 3)

        x = torch.matmul(x[:, :, 0].unsqueeze(3), x[:, :, 1].unsqueeze(
            2))  # (batch_size, num_points, 3, 1) * (batch_size, num_points, 1, 3) -> (batch_size, num_points, 3, 3)
        # x = torch.matmul(x[:,:,1:].transpose(3, 2), x[:,:,1:])
        x = x.view(batch_size, num_points, 9).transpose(2, 1)  # (batch_size, 9, num_points)

        x = torch.cat((pts, x), dim=1)  # (batch_size, 12, num_points)

        return x

    def graph_layer(self, x, idx):
        x = self.local_maxpool(x, idx)
        x = self.linear1(x)
        x = x.transpose(2, 1)
        x = F.relu(self.conv1(x))
        x = self.local_maxpool(x, idx)
        x = self.linear2(x)
        x = x.transpose(2, 1)
        x = self.conv2(x)
        return x

    def forward(self, x):
        x = x.transpose(2, 1)               # (batch_size, 3, num_points)
        idx = knn(x, k=self.k)
        x = self.local_cov(x, idx)            # (batch_size, 3, num_points) -> (batch_size, 12, num_points])
        x = self.mlp1(x)                        # (batch_size, 12, num_points) -> (batch_size, 64, num_points])
        x = self.graph_layer(x, idx)            # (batch_size, 64, num_points) -> (batch_size, 1024, num_points)
        x = torch.max(x, 2, keepdim=True)[0]    # (batch_size, 1024, num_points) -> (batch_size, 1024, 1)
        x = self.mlp2(x)                        # (batch_size, 1024, 1) -> (batch_size, feat_dims, 1)
        feat = x.transpose(2,1)                 # (batch_size, feat_dims, 1) -> (batch_size, 1, feat_dims)
        return feat                             # (batch_size, 1, feat_dims)




def get_encoder(encoder_name):
    dict_encoder = {
        'DGCNN_Vanilla': DGCNN_Vanilla,
        'DGCNN': DGCNN,
        'FoldingNet': FoldingNet,
    }
    return dict_encoder[encoder_name]
