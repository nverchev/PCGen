import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from encoder import Z_DIM, IN_CHAN, N_POINTS
from modules import PointsConvBlock, LinearBlock, View


class MLPDecoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.hdim = [256, 256]
        modules = [nn.Linear(Z_DIM, self.h_dim[0])]
        for i in range(len(self.h_dim) - 1):
            modules.append(nn.ELU())
            modules.append(nn.Linear(self.h_dim[i], self.h_dim[i + 1]))
        modules.append(nn.ELU())
        modules.append(nn.Linear(self.h_dim[-1], IN_CHAN * N_POINTS))
        modules.append(View(-1, N_POINTS, IN_CHAN))
        self.mlp = nn.Sequential(*modules)

    def forward(self, z):
        x = self.mlp(z)
        return x


class PointGenerator(nn.Module):

    def __init__(self):
        super().__init__()
        self.in_chan = IN_CHAN
        self.h_dim = [256, Z_DIM, 512, 256, 128, 64]
        self.m = 2048
        self.m_training = 128
        self.sample_dim = 16
        self.map_latent_mul1 = PointsConvBlock(self.sample_dim, self.h_dim[0])
        self.map_latent_mul2 = PointsConvBlock(self.h_dim[0], self.h_dim[1], act=nn.Hardtanh())
        modules = []
        for i in range(1, len(self.h_dim) - 1):
            modules.append(PointsConvBlock(self.h_dim[i], self.h_dim[i + 1]))
        modules.append(nn.Conv1d(self.h_dim[-1], IN_CHAN, kernel_size=1))
        self.mlp = nn.Sequential(*modules)

    def forward(self, z, s=None):
        batch = z.size()[0]
        device = z.device
        x = s if s is not None else torch.randn(batch, self.sample_dim, self.m).to(device)
        # x /= torch.linalg.vector_norm(x, dim=1, keepdim=True)
        x = self.map_latent_mul1(x)
        x = self.map_latent_mul2(x)
        x = z.unsqueeze(2) * x
        x = self.mlp(x)
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


class FoldingNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_chan = IN_CHAN
        # Sample the grids in 2D space
        num_grid = 45
        self.h_dim = [512] * 4
        self.m_grid = num_grid ** 2
        xx = torch.linspace(-0.3, 0.3, num_grid, dtype=torch.float)
        yy = torch.linspace(-0.3, 0.3, num_grid, dtype=torch.float)
        self.grid = torch.stack(torch.meshgrid(xx, yy, indexing="ij")).view(2, -1)  # (2, 45, 45) -> (2, 45 * 45)
        self.fold1 = FoldingLayer(Z_DIM + 2, [self.h_dim[0], self.h_dim[1], 3])
        self.fold2 = FoldingLayer(Z_DIM + 3, [self.h_dim[2], self.h_dim[3], 3])

    def forward(self, z):
        batch_size = z.shape[0]

        # repeat grid for batch operation
        grid = self.grid.to(z.device)  # (2, 45 * 45)
        grid = grid.unsqueeze(0).repeat(batch_size, 1, 1)  # (B, 2, 45 * 45)

        # repeat codewords
        x = z.unsqueeze(2).repeat(1, 1, self.m_grid)

        # two folding operations
        recon1 = self.fold1(grid, x)
        recon2 = self.fold2(recon1, x)
        return recon2.transpose(2, 1)


class FoldingLayer(nn.Module):
    """
    The folding operation of FoldingNet
    """

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


def get_decoder(decoder_name):
    decoder_dict = {
        "MLP": MLPDecoder,
        "Gen": PointGenerator,
        "FoldingNet": FoldingNet,
    }
    return decoder_dict[decoder_name]
