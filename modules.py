import torch
import torch.nn as nn


class Transpose(nn.Module):

    def forward(self, x):
        return x.transpose(2, 1)


class View(nn.Module):

    def __init__(self, *shape):
        super().__init__()
        self.shape = shape

    def forward(self, input):
        return input.view(self.shape)


class MaxperChannel(nn.Module):

    def forward(self, x):
        return torch.max(x, 1)[0]


# function pretending to be a class
def PointsBatchNorm(dim):
    return nn.Sequential(Transpose(), nn.BatchNorm1d(dim), Transpose())


class DBR(nn.Module):
    # Dense + Batch + Relu
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.dense = nn.Linear(in_dim, out_dim)
        self.bn = PointsBatchNorm(out_dim)
        self.relu = nn.ReLU()
        self.in_dim = in_dim
        self.out_dim = out_dim

    def forward(self, x):
        return self.relu(self.bn(self.dense(x)))


class DBR4(DBR):

    def forward(self, x, n_samles, m):
        x = x.view(-1, m, self.in_dim)
        x = super().forward(x)
        x = x.view(-1, n_samles, m, self.out_dim)
        return x


# Deals with features, no need of transposing
class DbR(nn.Module):
    # Dense + Batch + Relu
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.dense = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.dense(x)))


class STN(nn.Module):
    def __init__(self, channels=3):
        super().__init__()
        self.channels = channels
        self.net = nn.Sequential(DBR(channels, 64),
                                 DBR(64, 128),
                                 DBR(128, 1024),
                                 MaxperChannel(),
                                 DbR(1024, 512),
                                 DbR(512, 256),
                                 nn.Linear(256, channels ** 2))

        self.register_buffer('eye', torch.eye(channels))  # changes device automatically

    def forward(self, x):
        x = self.net(x).view(-1, self.channels, self.channels)
        x += self.eye
        return x


class PointNetEncoder(nn.Module):

    def __init__(self, in_chan, h_chan, z_dim):
        super().__init__()
        self.in_chan = in_chan
        self.h_chan = h_chan
        self.stn = STN()
        self.stnk = STN(h_chan[0])
        self.dbr1 = DBR(in_chan, h_chan[0])
        modules = [DBR(h_chan[0], h_chan[1]),
                   nn.Linear(h_chan[1], h_chan[2]),
                   PointsBatchNorm(h_chan[2]),
                   MaxperChannel(),
                   DbR(self.h_chan[2], self.h_chan[3]),
                   DbR(self.h_chan[3], 2 * z_dim)]
        self.encode = nn.Sequential(*modules)

    def forward(self, x):
        trans = self.stn(x)
        x = torch.matmul(x, trans)
        x = self.dbr1(x)
        trans = self.stnk(x)
        x = torch.matmul(x, trans)
        x = self.encode(x)
        return [x, trans]


class PointNetGenerator(nn.Module):

    def __init__(self, in_chan, h_chan, z_dim, m, n_samples=1):
        super().__init__()
        self.in_chan = in_chan
        self.h_chan = h_chan
        self.m_training = m
        self.m = m
        self.n_samples = n_samples
        self.sample_dim = 64
        self.h = 64
        self.hz = 256
        self.h2 = 128
        self.h3 = 256
        self.dbr = DBR4(self.sample_dim, self.h)
        self.map_latent1 = DbR(z_dim, self.hz)
        self.map_latent2 = DbR(self.hz, self.h ** 2)
        self.map_latent3 = DbR(self.hz, self.h ** 2)
        self.dbr1 = DBR4(self.h, self.h)
        self.dbr2 = DBR4(self.h, self.h3)

        self.lin = nn.Linear(self.h3, in_chan)
        self.register_buffer('eye2', torch.eye(self.h))

    def forward(self, z):
        batch = z.size()[0]
        device = z.device
        x = torch.randn(batch, self.n_samples, self.m, self.sample_dim).to(device)
        x /= torch.linalg.vector_norm(x, dim=2, keepdim=True)
        x = self.dbr(x, self.n_samples, self.m)
        z = self.map_latent1(z)
        trans = self.map_latent2(z)
        trans = trans.view(-1, self.h, self.h) + self.eye2
        trans = trans.unsqueeze(1).expand(-1, self.n_samples, -1, -1)
        x = torch.matmul(x, trans)
        x = self.dbr1(x, self.n_samples, self.m)
        trans = self.map_latent3(z)
        trans = trans.view(-1, self.h, self.h) + self.eye2
        trans = trans.unsqueeze(1).expand(-1, self.n_samples, -1, -1)
        x = torch.matmul(x, trans)
        x = self.dbr2(x, self.n_samples, self.m)
        x = self.lin(x)
        x = torch.tanh(x)
        x = x - x.mean(2, keepdim=True)
        return x.squeeze()
        return [x, trans]

    @property
    def m(self):
        if self.training:
            return self.m_training
        else:
            return self._m

    @m.setter
    def m(self, m):
        self._m = m


class PointGenerator(nn.Module):

    def __init__(self, in_chan, h_chan, z_dim, m, n_samples=1):
        super().__init__()
        self.in_chan = in_chan
        self.h_chan = h_chan
        self.m_training = m
        self.m = m
        self.n_samples = n_samples
        self.sample_dim = 8
        self.hz = 256
        self.h = 1024
        self.h2 = 256
        self.dbr = DBR4(self.sample_dim, self.h)
        self.map_latent = DbR(z_dim, self.hz)
        self.map_latent1 = nn.Linear(self.hz, self.h)
        self.dbr1 = DBR4(self.h, self.h2)
        self.dbr2 = DBR4(self.h2, self.h2)
        self.dbr3 = DBR4(self.h2, self.h2)
        self.lin = nn.Linear(self.h2, in_chan)

    def forward(self, z):
        batch = z.size()[0]
        device = z.device
        x = torch.rand(batch, self.n_samples, self.m, self.sample_dim).to(device)
        x = self.dbr(x, self.n_samples, self.m)
        z = self.map_latent(z)
        trans = torch.tanh(self.map_latent1(z))
        trans = trans.view(-1, 1, 1, self.h)
        trans = trans.expand(-1, self.n_samples, -1, -1)
        x = x * trans
        x = self.dbr1(x, self.n_samples, self.m)
        x = self.dbr2(x, self.n_samples, self.m)
        x = self.dbr3(x, self.n_samples, self.m)
        x = self.lin(x)
        x = torch.tanh(x)
        x = x - x.mean(2, keepdim=True)
        return x.squeeze()

    @property
    def m(self):
        if self.training:
            return self.m_training
        else:
            return self._m

    @m.setter
    def m(self, m):
        self._m = m


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean, scatter_max
from src.common import coordinate2index, normalize_coordinate, normalize_3d_coordinate, map2local



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
        self.bn7 = nn.BatchNorm1d(c_dim)

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
        self.conv7 = nn.Sequential(nn.Conv1d(512, c_dim, kernel_size=1, bias=False),
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

        return x.permute(0, 2, 1).contiguous()


class DGCNN_cls(nn.Module):
    '''
    Args:
        c_dim (int): dimension of latent code c
        scatter_type (str): feature aggregation when doing local pooling
        unet (bool): weather to use U-Net
        unet_kwargs (str): U-Net parameters
        unet3d (bool): weather to use 3D U-Net
        unet3d_kwargs (str): 3D U-Net parameters
        plane_resolution (int): defined resolution for plane feature
        grid_resolution (int): defined resolution for grid feature
        plane_type (str): feature type, 'xz' - 1-plane, ['xz', 'xy', 'yz'] - 3-plane, ['grid'] - 3D grid volume
        padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
    '''

    def __init__(self, c_dim=128, scatter_type='max',
                 unet=False, unet_kwargs=None, unet3d=False, unet3d_kwargs=None,
                 plane_resolution=None, grid_resolution=None, plane_type='xz', padding=0.1, feat_dim=1024, k=20):
        super().__init__()
        self.c_dim = c_dim

        self.dgcnn_encoder = DGCNN_Cls_Encoder(c_dim=self.c_dim, feat_dim=feat_dim, k=k)

        if unet:
            self.unet = UNet(c_dim, in_channels=c_dim, **unet_kwargs)
        else:
            self.unet = None

        if unet3d:
            self.unet3d = UNet3D(**unet3d_kwargs)
        else:
            self.unet3d = None

        self.reso_plane = plane_resolution
        self.reso_grid = grid_resolution
        self.plane_type = plane_type
        self.padding = padding

        if scatter_type == 'max':
            self.scatter = scatter_max
        elif scatter_type == 'mean':
            self.scatter = scatter_mean
        else:
            raise ValueError('incorrect scatter type')

    def generate_plane_features(self, p, c, plane='xz'):
        # acquire indices of features in plane
        xy = normalize_coordinate(p.clone(), plane=plane, padding=self.padding)  # normalize to the range of (0, 1)
        index = coordinate2index(xy, self.reso_plane)

        # scatter plane features from points
        fea_plane = c.new_zeros(p.size(0), self.c_dim, self.reso_plane ** 2)
        c = c.permute(0, 2, 1)  # B x 512 x T
        fea_plane = scatter_mean(c, index, out=fea_plane)  # B x 512 x reso^2
        fea_plane = fea_plane.reshape(p.size(0), self.c_dim, self.reso_plane,
                                      self.reso_plane)  # sparce matrix (B x 512 x reso x reso)

        # process the plane features with UNet
        if self.unet is not None:
            fea_plane = self.unet(fea_plane)

        return fea_plane

    def generate_grid_features(self, p, c):
        p_nor = normalize_3d_coordinate(p.clone(), padding=self.padding)
        index = coordinate2index(p_nor, self.reso_grid, coord_type='3d')
        # scatter grid features from points
        fea_grid = c.new_zeros(p.size(0), self.c_dim, self.reso_grid ** 3)
        c = c.permute(0, 2, 1)
        fea_grid = scatter_mean(c, index, out=fea_grid)  # B x C x reso^3
        fea_grid = fea_grid.reshape(p.size(0), self.c_dim, self.reso_grid, self.reso_grid,
                                    self.reso_grid)  # sparce matrix (B x 512 x reso x reso)

        if self.unet3d is not None:
            fea_grid = self.unet3d(fea_grid)

        return fea_grid

    def pool_local(self, xy, index, c):
        bs, fea_dim = c.size(0), c.size(2)
        keys = xy.keys()

        c_out = 0
        for key in keys:
            # scatter plane features from points
            if key == 'grid':
                fea = self.scatter(c.permute(0, 2, 1), index[key], dim_size=self.reso_grid ** 3)
            else:
                fea = self.scatter(c.permute(0, 2, 1), index[key], dim_size=self.reso_plane ** 2)
            if self.scatter == scatter_max:
                fea = fea[0]
            # gather feature back to points
            fea = fea.gather(dim=2, index=index[key].expand(-1, fea_dim, -1))
            c_out += fea
        return c_out.permute(0, 2, 1)

    def forward(self, p):
        batch_size, T, D = p.size()

        c = self.dgcnn_encoder(p)

        fea = {}
        if 'grid' in self.plane_type:
            fea['grid'] = self.generate_grid_features(p, c)
        if 'xz' in self.plane_type:
            fea['xz'] = self.generate_plane_features(p, c, plane='xz')
        if 'xy' in self.plane_type:
            fea['xy'] = self.generate_plane_features(p, c, plane='xy')
        if 'yz' in self.plane_type:
            fea['yz'] = self.generate_plane_features(p, c, plane='yz')

        return fea

