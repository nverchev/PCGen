import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from src.modules import PointsConvBlock, LinearBlock, View
from src.utils import graph_max_pooling
CHAN_OUT = 3

class MLPDecoder(nn.Module):

    def __init__(self, z_dim, m):
        super().__init__()
        self.hdim = [256, 256]
        modules = [nn.Linear(z_dim, self.h_dim[0])]
        for i in range(len(self.h_dim) - 1):
            modules.append(nn.ELU())
            modules.append(nn.Linear(self.h_dim[i], self.h_dim[i + 1]))
        modules.append(nn.ELU())
        modules.append(nn.Linear(self.h_dim[-1], CHAN_OUT * m))
        modules.append(View(-1, m, CHAN_OUT))
        self.mlp = nn.Sequential(*modules)

    def forward(self, z):
        x = self.mlp(z)
        return x


class PCGen(nn.Module):

    def __init__(self, z_dim, m):
        super().__init__()
        self.h_dim = [256, z_dim, 512, 256, 128, 64]
        self.m = 2048
        self.m_training = m
        self.sample_dim = 16
        self.map_samples1 = nn.Conv1d(self.sample_dim, self.h_dim[0], 1)
        self.map_samples2 = nn.Conv1d(self.h_dim[0], self.h_dim[1], 1)
        modules = []
        for i in range(1, len(self.h_dim) - 1):
            modules.append(PointsConvBlock(self.h_dim[i], self.h_dim[i + 1]))
        modules.append(nn.Conv1d(self.h_dim[-1], CHAN_OUT, kernel_size=1))
        self.points_convs = nn.Sequential(*modules)

    def forward(self, z, s=None):
        batch = z.size()[0]
        device = z.device
        x = s if s is not None else torch.randn(batch, self.sample_dim, self.m, device=device)
        # x /= torch.linalg.vector_norm(x, dim=1, keepdim=True)
        x = F.leaky_relu_(self.map_samples1(x))
        x = F.hardtanh_(self.map_samples2(x))
        x = z.unsqueeze(2) * x
        x = self.points_convs(x)
        return x.transpose(2, 1)
    @property
    def m(self):
        if self.training:
            return self.m_training
        else:
            return self._m

    @m.setter
    def m(self, m):
        self._m = m

class Gen_ADAIN(nn.Module):

    def __init__(self, z_dim, m):
        super().__init__()
        self.h_dim = [z_dim] * 5
        self.m = 2048
        self.m_training = m
        self.sample_dim = z_dim
        map_samples = []
        map_latents = []
        for i in range(len(self.h_dim) - 1):
            map_samples.append(PointsConvBlock(self.h_dim[i], self.h_dim[i+1]))
            map_latents.append(PointsConvBlock(self.h_dim[i], 2 * self.h_dim[i+1]))
        self.map_samples = nn.ModuleList(map_samples)
        self.map_latents = nn.ModuleList(map_latents)
        self.final = nn.Conv1d(self.h_dim[-1], CHAN_OUT, kernel_size=1)

    def forward(self, z, s=None):
        batch = z.size()[0]
        device = z.device
        x = s if s is not None else torch.randn(batch, self.sample_dim, self.m).to(device)
        z = z.unsqueeze(2)
        for map_sample, map_latent in zip(self.map_samples, self.map_latents):
            x = map_sample(x)
            y_mean, y_bias = map_latent(z).chunk(2, 1)
            x = y_mean * (x - x.mean(2, keepdim=True) / x.var(2, keepdim=True)) + y_bias
        x = self.final(x)
        return x.transpose(2, 1)


class FoldingLayer(nn.Module):
    '''
    The folding operation of FoldingNet
    '''

    def __init__(self, in_channel: int, out_channels: list):
        super(FoldingLayer, self).__init__()

        modules = []
        for oc in out_channels[:-1]:
            modules.append(PointsConvBlock(in_channel, oc, act=nn.ReLU()))
            in_channel = oc
        out_layer = nn.Conv1d(in_channel, out_channels[-1], 1)
        modules.append(out_layer)
        self.layers = nn.Sequential(*modules)

    def forward(self, grids, x):
        # concatenate
        x = torch.cat([grids, x], dim=1)
        # shared mlp
        x = self.layers(x)
        return x


class FoldingNet(nn.Module):
    def __init__(self, z_dim, m):
        super().__init__()
        # Sample the grids in 2D space
        self.num_grid = round(np.sqrt(m))
        self.z_dim = z_dim
        self.h_dim = [512] * 4
        self.m_grid = self.num_grid ** 2
        xx = torch.linspace(-0.3, 0.3, self.num_grid, dtype=torch.float)
        yy = torch.linspace(-0.3, 0.3, self.num_grid, dtype=torch.float)
        self.grid = nn.Parameter(torch.stack(torch.meshgrid(xx, yy, indexing='ij')).view(2, -1), requires_grad=False)
        self.fold1 = FoldingLayer(z_dim + 2, [self.h_dim[0], self.h_dim[1], CHAN_OUT])
        self.fold2 = FoldingLayer(z_dim + 3, [self.h_dim[2], self.h_dim[3], CHAN_OUT])

    def forward(self, z, grid=None):
        batch_size = z.shape[0]

        # repeat grid for batch operation
        if grid is None:
            grid = self.grid.unsqueeze(0).repeat(batch_size, 1, 1)  # (B, 2, 45 * 45)

        # repeat codewords
        x = z.unsqueeze(2).repeat(1, 1, self.m_grid)
        # two folding operations
        recon1 = self.fold1(grid, x)
        recon2 = self.fold2(recon1, x)
        return recon2.transpose(2, 1)


class TearingNetGraphModel(FoldingNet):
    def __init__(self, z_dim, m):
        super().__init__(z_dim, m)
        self.graph_r = 1e-12
        self.graph_eps = 0.02
        self.graph_eps_sqr = self.graph_eps ** 2
        self.graph_lam = 0.5
        self.h_dim.extend([512, 512, 64, 512, 512, 2])

        modules = [nn.Conv2d(self.z_dim + 5, self.h_dim[4], kernel_size=1)]
        for i in range(4, 6):
            modules.append(nn.ReLU())
            modules.append(nn.Conv2d(self.h_dim[i], self.h_dim[i+1], kernel_size=1))
        self.tearing1 = nn.Sequential(*modules)

        modules = [nn.Conv2d(self.z_dim + 5 + self.h_dim[6], self.h_dim[7], kernel_size=1)]
        for i in range(7, 9):
            modules.append(nn.ReLU())
            modules.append(nn.Conv2d(self.h_dim[i], self.h_dim[i+1], kernel_size=1))
        self.tearing2 = nn.Sequential(*modules)
    def forward(self, z, grid=None):
        batch_size = z.shape[0]
        grid = self.grid.to(z.device)
        grid = grid.unsqueeze(0).repeat(batch_size, 1, 1)
        x = super().forward(z, grid)
        grid_exp = grid.view(batch_size, 2, self.num_grid, self.num_grid)
        x_exp = x.transpose(2, 1).view(-1, 3, self.num_grid, self.num_grid)
        z_exp = z.view(-1, self.z_dim, 1, 1).expand(-1, -1, self.num_grid, self.num_grid)
        in1 = torch.cat((grid_exp, x_exp, z_exp), 1).contiguous()
        # Compute the torn 2D grid
        out1 = self.tearing1(in1)  # 1st tearing
        in2 = torch.cat((in1, out1), 1)
        out2 = self.tearing2(in2)  # 2nd tearing
        out2 = out2.contiguous().view(batch_size, 2, self.num_grid * self.num_grid)
        grid = grid + out2
        pc1 = super().forward(z, grid)
        # pc2, graph_wght = self.graph_filter(pc1, grid, batch_size)  # Graph Filtering
        return pc1

    # def graph_filter(self, pc, grid, batch_size):
    #     grid_exp = grid.view(batch_size, 2, self.num_grid, self.num_grid)
    #     pc_exp = pc.transpose(2, 1).view(-1, 3, self.num_grid, self.num_grid)
    #     graph_feature = torch.cat((grid_exp, pc_exp), 1).contiguous()
    #     # Compute the graph weights
    #     wght_hori = graph_feature[:, :, :-1, :] - graph_feature[:, :, 1:, :]  # horizontal weights
    #     wght_vert = graph_feature[:, :, :, :-1] - graph_feature[:, :, :, 1:]  # vertical weights
    #     wght_hori = torch.exp(-torch.sum(wght_hori * wght_hori, dim=1) / self.graph_eps_sqr)  # Gaussian weight
    #     wght_vert = torch.exp(-torch.sum(wght_vert * wght_vert, dim=1) / self.graph_eps_sqr)
    #     wght_hori = (wght_hori > self.graph_r) * wght_hori
    #     wght_vert = (wght_vert > self.graph_r) * wght_vert
    #     wght_lft = torch.cat((torch.zeros([batch_size, 1, self.num_grid]).cuda(), wght_hori), 1)  # add left
    #     wght_rgh = torch.cat((wght_hori, torch.zeros([batch_size, 1, self.num_grid]).cuda()), 1)  # add right
    #     wght_top = torch.cat((torch.zeros([batch_size, self.num_grid, 1]).cuda(), wght_vert), 2)  # add top
    #     wght_bot = torch.cat((wght_vert, torch.zeros([batch_size, self.num_grid, 1]).cuda()), 2)  # add bottom
    #     wght_all = torch.cat(
    #         (wght_lft.unsqueeze(1), wght_rgh.unsqueeze(1), wght_top.unsqueeze(1), wght_bot.unsqueeze(1)), 1)
    #
    #     # Perform the actural graph filtering: x = (I - \lambda L) * x
    #     wght_hori = wght_hori.unsqueeze(1).expand(-1, 3, -1, -1)  # dimension expansion
    #     wght_vert = wght_vert.unsqueeze(1).expand(-1, 3, -1, -1)
    #     pc = pc.permute([0, 2, 1]).contiguous().view(batch_size, 3, self.num_grid, self.num_grid)
    #     pc_filt = \
    #         torch.cat((torch.zeros([batch_size, 3, 1, self.num_grid]).cuda(), pc[:, :, :-1, :] * wght_hori), 2) + \
    #         torch.cat((pc[:, :, 1:, :] * wght_hori, torch.zeros([batch_size, 3, 1, self.num_grid]).cuda()), 2) + \
    #         torch.cat((torch.zeros([batch_size, 3, self.num_grid, 1]).cuda(), pc[:, :, :, :-1] * wght_vert), 3) + \
    #         torch.cat((pc[:, :, :, 1:] * wght_vert, torch.zeros([batch_size, 3, self.num_grid, 1]).cuda()),
    #                   3)  # left, right, top, bottom
    #
    #     pc_filt = pc + self.graph_lam * (pc_filt - torch.sum(wght_all, dim=1).unsqueeze(1).expand(-1, 3, -1,
    #                                                                                               -1) * pc)  # equivalent to ( I - \lambda L) * x
    #     pc_filt = pc_filt.view(batch_size, 3, -1).permute([0, 2, 1])
    #     return pc_filt, wght_all

def get_decoder(decoder_name):
    decoder_dict = {
        'MLP': MLPDecoder,
        'PCGen': PCGen,
        'AdaIN': Gen_ADAIN,
        'FoldingNet': FoldingNet,
        'TearingNet': TearingNetGraphModel,
    }
    return decoder_dict[decoder_name]
