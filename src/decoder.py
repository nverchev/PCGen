import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from src.modules import PointsConvBlock, LinearBlock, View
from src.neighbour_op import get_neighbours
OUT_CHAN = 3


# class MLPDecoder(nn.Module):
#
#     def __init__(self, z_dim, m):
#         super().__init__()
#         self.hdim = [256, 256, 256, 256]
#         modules = [nn.Linear(z_dim, self.h_dim[0])]
#         for i in range(len(self.h_dim) - 1):
#             modules.append(nn.ELU())
#             modules.append(nn.Linear(self.h_dim[i], self.h_dim[i + 1]))
#         modules.append(nn.ELU())
#         modules.append(nn.Linear(self.h_dim[-1], OUT_CHAN * m))
#         modules.append(View(-1, m, OUT_CHAN))
#         self.mlp = nn.Sequential(*modules)
#
#     def forward(self, z):
#         x = self.mlp(z)
#         return x


class PCGen(nn.Module):

    def __init__(self, z_dim, m, gf=True):
        super().__init__()
        self.h_dim = [256, z_dim, 512, 512, 512, 64]
        self.m = 2048
        self.m_training = m
        self.gf = gf
        self.sample_dim = 16
        self.map_samples1 = PointsConvBlock(self.sample_dim, self.h_dim[0], batch_norm=False)
        self.map_samples2 = PointsConvBlock(self.h_dim[0], self.h_dim[1], batch_norm=False, act=nn.Hardtanh())
        modules = []
        for in_dim, out_dim in zip(self.h_dim[1:-1], self.h_dim[2:]):
            modules.append(PointsConvBlock(in_dim, out_dim, batch_norm=False))

        modules.append(PointsConvBlock(self.h_dim[-1], OUT_CHAN, act=None, batch_norm=False))
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
            x = self.graph_filtering(x)
        return x

    def graph_filtering(self, x):
        dist, neighbours = get_neighbours(x, k=4, indices=None)
        dist1 = dist[..., 1:]  # dist[:, :,  0] == 0
        neighbours1 = neighbours[..., 1:]
        sigma2 = torch.sqrt(dist1.mean(-1, keepdims=True))
        weights = torch.softmax(-dist1 / sigma2, dim=-1)
        weighted_neighbours = weights.unsqueeze(1).expand(-1,  3, -1, -1) * neighbours1
        x = 2 * x - 1 * weighted_neighbours.sum(-1)
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


#
# class Gen_ADAIN(nn.Module):
#
#     def __init__(self, z_dim, m):
#         super().__init__()
#         self.h_dim = [z_dim] * 5
#         self.m = 2048
#         self.m_training = m
#         self.sample_dim = z_dim
#         map_samples = []
#         map_latents = []
#         for i in range(len(self.h_dim) - 1):
#             map_samples.append((self.h_dim[i], self.h_dim[i + 1]))
#             map_latents.append(PointsConvBlock(self.h_dim[i], 2 * self.h_dim[i + 1]))
#         self.map_samples = nn.ModuleList(map_samples)
#         self.map_latents = nn.ModuleList(map_latents)
#         self.final = nn.Conv1d(self.h_dim[-1], OUT_CHAN, kernel_size=1)
#
#     def forward(self, z, s=None):
#         batch = z.size()[0]
#         device = z.device
#         x = s if s is not None else torch.randn(batch, self.sample_dim, self.m).to(device)
#         z = z.unsqueeze(2)
#         for map_sample, map_latent in zip(self.map_samples, self.map_latents):
#             x = map_sample(x)
#             y_mean, y_bias = map_latent(z).chunk(2, 1)
#             x = y_mean * (x - x.mean(2, keepdim=True) / x.var(2, keepdim=True)) + y_bias
#         x = self.final(x)
#         return x.transpose(2, 1)


class FoldingLayer(nn.Module):
    '''
    The folding operation of FoldingNet
    '''

    def __init__(self, in_channel: int, h_dim: list):
        super(FoldingLayer, self).__init__()
        modules = [nn.Conv1d(in_channel, h_dim[0], kernel_size=1)]
        for in_dim, out_dim in zip(h_dim[0:-1], h_dim[1:]):
            modules.extend([nn.ReLU(), nn.Conv1d(in_dim, out_dim, kernel_size=1)])
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
        wght_hori = graph_feature[:, :, :-1, :] - graph_feature[:, :, 1:, :]  # horizontal weights
        wght_vert = graph_feature[:, :, :, :-1] - graph_feature[:, :, :, 1:]  # vertical weights
        wght_hori = torch.exp(-torch.sum(wght_hori ** 2, dim=1) / self.graph_eps_sqr)  # Gaussian weight
        wght_vert = torch.exp(-torch.sum(wght_vert ** 2, dim=1) / self.graph_eps_sqr)
        wght_hori = (wght_hori > self.graph_r) * wght_hori
        wght_vert = (wght_vert > self.graph_r) * wght_vert

        wght_lft = F.pad(wght_hori, pad=[0, 0, 1, 0])
        wght_rgh = F.pad(wght_hori, pad=[0, 0, 0, 1])
        wght_top = F.pad(wght_vert, pad=[1, 0])
        wght_bot = F.pad(wght_vert, pad=[0, 1])


        D = torch.stack((wght_lft, wght_rgh, wght_top, wght_bot), dim=1)
        D = torch.sum(D, dim=1).unsqueeze(1).expand(-1, 3, -1, -1)
        wght_hori = wght_hori.unsqueeze(1).expand(-1, 3, -1, -1)
        wght_vert = wght_vert.unsqueeze(1).expand(-1, 3, -1, -1)
        pc_filt1 = F.pad((pc_exp[:, :, :-1, :] * wght_hori), [0, 0, 1, 0])
        pc_filt1 += F.pad((pc_exp[:, :, 1:, :] * wght_hori), [0, 0, 0, 1])
        pc_filt1 += F.pad((pc_exp[:, :, :, :-1] * wght_vert), [1, 0])
        pc_filt1 += F.pad((pc_exp[:, :, :, 1:] * wght_vert), [0, 1])
        pc_filt = (1 - self.graph_lam * D) * pc_exp + self.graph_lam * pc_filt1
        pc_filt = pc_filt.view(batch_size, 3, -1)
        return pc_filt


class TearingNet(FoldingNet):
    def __init__(self, z_dim, m, gf):
        super().__init__(z_dim, m, gf=False)
        self.h_dim.extend([512, 512, 64, 512, 512, 2])

        modules = [nn.Conv2d(self.z_dim + 5, self.h_dim[4], kernel_size=1)]
        for in_dim, out_dim in zip(self.h_dim[4:6], self.h_dim[5:7]):
            modules.append(nn.ReLU())
            modules.append(nn.Conv2d(in_dim, out_dim, kernel_size=1))
        self.tearing1 = nn.Sequential(*modules)

        modules = [nn.Conv2d(self.z_dim + 5 + self.h_dim[6], self.h_dim[7], kernel_size=1)]
        for in_dim, out_dim in zip(self.h_dim[7:9], self.h_dim[8:10]):
            modules.append(nn.ReLU())
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
            x, graph_wght = self.graph_filter(x, grid, batch_size)
        return x


def get_decoder(decoder_name):
    decoder_dict = {
        # 'MLP': MLPDecoder,
        'PCGen': PCGen,
        # 'AdaIN': Gen_ADAIN,
        'FoldingNet': FoldingNet,
        'TearingNet': TearingNet,
    }
    return decoder_dict[decoder_name]
