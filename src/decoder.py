import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from src.modules import PointsConvBlock, LinearBlock
from src.neighbour_op import graph_filtering

OUT_CHAN = 3


class FullyConnected(nn.Module):

    def __init__(self, z_dim, m, gf):
        super().__init__()
        self.z_dim = z_dim
        self.h_dim = [256] * 2
        self.gf = gf
        self.m = m
        modules = [LinearBlock(z_dim, self.h_dim[0], batch_norm=False, act=nn.ReLU(inplace=True)),
                   LinearBlock(self.h_dim[0], self.h_dim[1], batch_norm=False, act=nn.ReLU(inplace=True)),
                   LinearBlock(self.h_dim[1], OUT_CHAN * m, batch_norm=False, act=None)]
        self.mlp = nn.Sequential(*modules)

    def forward(self, z):
        x = self.mlp(z)
        return x.view(-1, OUT_CHAN, self.m)

class FoldingLayer(nn.Module):
    """
    The folding operation of FoldingNet
    """

    def __init__(self, in_channel: int, h_dim: list):
        super(FoldingLayer, self).__init__()
        modules = [nn.Conv1d(in_channel, h_dim[0], kernel_size=1)]
        for in_dim, out_dim in zip(h_dim[0:-1], h_dim[1:]):
            modules.extend([nn.ReLU(inplace=True), nn.Conv1d(in_dim, out_dim, kernel_size=1)])
        self.layers = nn.Sequential(*modules)

    def forward(self, grids, x):
        x = torch.cat([grids, x], dim=1).contiguous()
        x = self.layers(x)
        return x


class FoldingNet(nn.Module):
    def __init__(self, z_dim, m, gf):
        super().__init__()
        self.z_dim = z_dim
        self.h_dim = [512] * 4
        self.gf = gf
        self.m = m
        self.num_grid = round(np.sqrt(m))
        self.m_grid = self.num_grid ** 2
        xx = torch.linspace(-0.3, 0.3, self.num_grid, dtype=torch.float)
        yy = torch.linspace(-0.3, 0.3, self.num_grid, dtype=torch.float)
        self.grid = nn.Parameter(torch.stack(torch.meshgrid(xx, yy, indexing='ij')).view(2, -1), requires_grad=False)
        self.fold1 = FoldingLayer(z_dim + 2, self.h_dim[0:2] + [OUT_CHAN])
        self.fold2 = FoldingLayer(z_dim + 3, self.h_dim[2:4] + [OUT_CHAN])
        self.graph_r = 1e-12
        self.graph_eps = 0.02
        self.graph_eps_sqr = self.graph_eps ** 2
        self.graph_lam = 0.5

    def forward(self, z, grid=None):
        batch_size = z.shape[0]
        if grid is None:
            grid = self.grid.unsqueeze(0).repeat(batch_size, 1, 1)
        z = z.unsqueeze(2).repeat(1, 1, self.m_grid)
        x = self.fold1(grid, z)
        x = self.fold2(x, z)
        if self.gf:
            x = self.graph_filter(x, grid, batch_size)
        return x

    def graph_filter(self, pc, grid, batch_size):
        grid_exp = grid.view(batch_size, 2, self.num_grid, self.num_grid)
        pc_exp = pc.view(batch_size, 3, self.num_grid, self.num_grid)
        graph_feature = torch.cat((grid_exp, pc_exp), 1).contiguous()

        # Compute the graph weights
        delta_x = graph_feature[:, :, :-1, :] - graph_feature[:, :, 1:, :]  # horizontal weights
        delta_y = graph_feature[:, :, :, :-1] - graph_feature[:, :, :, 1:]  # vertical weights
        delta_x = torch.exp(-torch.sum(delta_x ** 2, dim=1) / self.graph_eps_sqr)  # Gaussian weight
        delta_y = torch.exp(-torch.sum(delta_y ** 2, dim=1) / self.graph_eps_sqr)
        delta_x = (delta_x > self.graph_r) * delta_x
        delta_y = (delta_y > self.graph_r) * delta_y

        delta_x_left = F.pad(delta_x, pad=[0, 0, 1, 0])
        delta_x_right = F.pad(delta_x, pad=[0, 0, 0, 1])
        delta_y_top = F.pad(delta_y, pad=[1, 0])
        delta_y_bottom = F.pad(delta_y, pad=[0, 1])

        delta = torch.stack((delta_x_left, delta_x_right, delta_y_top, delta_y_bottom), dim=1)
        delta = torch.sum(delta, dim=1).unsqueeze(1).expand(-1, 3, -1, -1)
        delta_x = delta_x.unsqueeze(1).expand(-1, 3, -1, -1)
        delta_y = delta_y.unsqueeze(1).expand(-1, 3, -1, -1)
        pc_filt1 = F.pad((pc_exp[:, :, :-1, :] * delta_x), [0, 0, 1, 0])
        pc_filt1 += F.pad((pc_exp[:, :, 1:, :] * delta_x), [0, 0, 0, 1])
        pc_filt1 += F.pad((pc_exp[:, :, :, :-1] * delta_y), [1, 0])
        pc_filt1 += F.pad((pc_exp[:, :, :, 1:] * delta_y), [0, 1])
        pc_filt = (1 - self.graph_lam * delta) * pc_exp + self.graph_lam * pc_filt1
        pc_filt = pc_filt.view(batch_size, 3, -1)
        return pc_filt


class TearingNet(FoldingNet):
    def __init__(self, z_dim, m, gf):
        super().__init__(z_dim, m, gf=gf)
        self.h_dim.extend([512, 512, 64, 512, 512, 2])
        modules = [nn.Conv2d(self.z_dim + 5, self.h_dim[4], kernel_size=1)]
        for in_dim, out_dim in zip(self.h_dim[4:6], self.h_dim[5:7]):
            modules.append(nn.ReLU(inplace=True))
            modules.append(nn.Conv2d(in_dim, out_dim, kernel_size=1))
        self.tearing1 = nn.Sequential(*modules)

        modules = [nn.Conv2d(self.z_dim + 5 + self.h_dim[6], self.h_dim[7], kernel_size=1)]
        for in_dim, out_dim in zip(self.h_dim[7:9], self.h_dim[8:10]):
            modules.append(nn.ReLU(inplace=True))
            modules.append(nn.Conv2d(in_dim, out_dim, kernel_size=1))
        self.tearing2 = nn.Sequential(*modules)

    def forward(self, z, grid=None):
        batch_size = z.shape[0]
        grid = self.grid.to(z.device)
        grid = grid.unsqueeze(0).repeat(batch_size, 1, 1)
        x = super().forward(z, grid)
        grid_exp = grid.view(batch_size, 2, self.num_grid, self.num_grid)
        x_exp = x.view(-1, 3, self.num_grid, self.num_grid)
        z_exp = z.view(-1, self.z_dim, 1, 1).expand(-1, -1, self.num_grid, self.num_grid)
        in1 = torch.cat((grid_exp, x_exp, z_exp), 1).contiguous()
        # Compute the torn 2D grid
        out1 = self.tearing1(in1)  # 1st tearing
        in2 = torch.cat((in1, out1), 1)
        out2 = self.tearing2(in2)  # 2nd tearing
        out2 = out2.contiguous().view(batch_size, 2, self.num_grid * self.num_grid)
        grid = grid + out2
        x = super().forward(z, grid)
        if self.gf:
            x = self.graph_filter(x, grid, batch_size)
        return x


# AtlasNet
class MLPAdj(nn.Module):
    def __init__(self, dim):
        """Atlas decoder"""

        super().__init__()
        self.h_dim = [dim, dim // 2, dim // 4]

        modules = [PointsConvBlock(dim, self.h_dim[0], act=nn.ReLU(inplace=True))]
        for in_dim, out_dim in zip(self.h_dim[0:-1], self.h_dim[1:]):
            modules.append(PointsConvBlock(in_dim, out_dim, act=nn.ReLU(inplace=True)))
        modules.append(PointsConvBlock(self.h_dim[-1], OUT_CHAN, batch_norm=False, act=nn.Tanh()))
        self.layers = nn.Sequential(*modules)

    def forward(self, x):
        return self.layers(x)


class PatchDeformationMLP(nn.Module):
    """deformation of a 2D patch into a 3D surface"""

    def __init__(self, patchDeformDim=3, h_dim=128):
        super().__init__()
        self.h_dim = h_dim
        modules = [PointsConvBlock(2, self.h_dim, act=nn.ReLU(inplace=True)),
                   PointsConvBlock(self.h_dim, self.h_dim, act=nn.ReLU(inplace=True)),
                   PointsConvBlock(self.h_dim, patchDeformDim, batch_norm=False, act=nn.Tanh())]
        self.layers = nn.Sequential(*modules)

    def forward(self, x):
        return self.layers(x)


class AtlasNetv2(nn.Module):
    """Atlas net PatchDeformMLPAdj"""

    def __init__(self, z_dim, m, gf):
        super().__init__()
        self.z_dim = z_dim
        self.m = m
        self.npatch = 8
        self.patchDeformDim = 10
        self.h_dim = 128
        self.patchDeformation = nn.ModuleList(
            PatchDeformationMLP(patchDeformDim=self.patchDeformDim, h_dim=self.h_dim) for _ in range(self.npatch))
        self.decoder = nn.ModuleList([MLPAdj(dim=self.patchDeformDim + self.z_dim) for _ in range(self.npatch)])

    def forward(self, x):
        batch = x.size(0)
        device = x.device
        outs = []
        for i in range(0, self.npatch):
            m_patch = self.m // self.npatch
            rand_grid = torch.rand(batch, 2, m_patch, device=device)
            rand_grid = self.patchDeformation[i](rand_grid)
            y = x.unsqueeze(2).expand(-1, -1, m_patch).contiguous()
            y = torch.cat((rand_grid, y), 1).contiguous()
            outs.append(self.decoder[i](y))

        return torch.cat(outs, 2)


class AtlasNetv2(nn.Module):
    """Atlas net PatchDeformMLPAdj"""

    def __init__(self, z_dim, m, gf):
        super().__init__()
        self.z_dim = z_dim
        self.m = m
        self.gf = gf
        self.num_patch = 8
        self.m_patch = self.m // self.num_patch
        self.deform_patch_dim = 10
        self.h_dims = [128]

        total_in = 2 * self.num_patch
        dim = self.h_dims[0] * self.num_patch
        total_out = self.deform_patch_dim * self.num_patch
        modules = [PointsConvBlock(total_in, dim, act=nn.ReLU(inplace=True), groups=self.num_patch),
                   PointsConvBlock(dim, dim, act=nn.ReLU(inplace=True), groups=self.num_patch),
                   PointsConvBlock(dim, total_out, batch_norm=False, act=nn.Tanh(), groups=self.num_patch)]
        self.patchDeformation = nn.Sequential(*modules)

        dim = (self.deform_patch_dim + self.z_dim) * self.num_patch
        self.dims = [dim, dim // 2, dim // 4]
        modules = [PointsConvBlock(dim, dim, act=nn.ReLU(inplace=True), groups=self.num_patch)]
        for in_dim, out_dim in zip(self.adj_dims[0:-1], self.dims[1:]):
            modules.append(PointsConvBlock(in_dim, out_dim, act=nn.ReLU(inplace=True), groups=self.num_patch))
        modules.append(PointsConvBlock(self.dims[-1], OUT_CHAN, batch_norm=False, act=nn.Tanh(), groups=self.num_patch))
        self.mlp_adj = nn.Sequential(*modules)

    def forward(self, z):
        batch = z.size(0)
        device = z.device
        x = z.unsqueeze(2).expand(-1, -1, self.num_patch * self.m_patch)
        x = x.view(-1, self.z_dim * self.num_patch, self.m_patch).contiguous()
        rand_grid = torch.rand(batch, 2 * self.num_patch, self.m_patch, device=device)
        deformed_grid = self.patchDeformation(rand_grid).view(-1, self.num_patch, self.deform_patch_dim, self.m_patch)
        x = torch.cat([deformed_grid, x.unsqueeze(2)], dim=2).view(batch, -1, self.m_patch).contiguous()
        x = self.mlp_adj(x).view(-1, OUT_CHAN, self.m)
        if self.gf:
            x = graph_filtering(x)
        return x


class PCGen(nn.Module):

    def __init__(self, z_dim, m, gf=True):
        super().__init__()
        self.h_dim = [256, z_dim, 512, 256, 128, 64]
        self.m = 2048
        self.m_training = m
        self.gf = gf
        self.sample_dim = 16
        self.map_samples1 = PointsConvBlock(self.sample_dim, self.h_dim[0], batch_norm=False, act=nn.ReLU(inplace=True))
        self.map_samples2 = PointsConvBlock(self.h_dim[0], self.h_dim[1], batch_norm=False,
                                            act=nn.Hardtanh(inplace=True))
        modules = []
        for in_dim, out_dim in zip(self.h_dim[1:-1], self.h_dim[2:]):
            modules.append(PointsConvBlock(in_dim, out_dim, batch_norm=False, act=nn.ReLU(inplace=True)))

        modules.append(PointsConvBlock(self.h_dim[-1], OUT_CHAN, batch_norm=False, act=None))
        self.points_convs = nn.Sequential(*modules)

    def forward(self, z, s=None):
        batch = z.size()[0]
        device = z.device
        x = s if s is not None else torch.randn(batch, self.sample_dim, self.m, device=device)
        x = self.map_samples1(x)
        x = self.map_samples2(x)
        x = z.unsqueeze(2) * x
        x = self.points_convs(x)
        if self.gf:
            x = graph_filtering(x)
        return x

    @property
    def m(self):
        if self.training:
            return self.m_training
        else:
            return self._m

    @m.setter
    def m(self, m):
        self._m = m
def get_decoder(decoder_name):
    decoder_dict = {
        'Full': FullyConnected,
        'FoldingNet': FoldingNet,
        'TearingNet': TearingNet,
        'AtlasNet': AtlasNetv2,
        'PCGen': PCGen,
    }
    return decoder_dict[decoder_name]
