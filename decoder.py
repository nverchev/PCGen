import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from encoder import Z_DIM, IN_CHAN, N_POINTS
from modules import PointsConvBlock, LinearBlock, PointsConvBlock4, View


class MLPDecoder(nn.Module):

    def __init__(self):
        super().__init__()
        h_dim = [256, 256]
        modules = [nn.Linear(Z_DIM, h_dim[0])]
        for i in range(len(h_dim) - 1):
            modules.append(nn.ELU())
            modules.append(nn.Linear(h_dim[i], h_dim[i + 1]))
        modules.append(nn.ELU())
        modules.append(nn.Linear(h_dim[-1], IN_CHAN * N_POINTS))
        modules.append(View(-1, N_POINTS, IN_CHAN))
        self.mlp = nn.Sequential(*modules)

    def forward(self, z):
        x = self.mlp(z)
        return x


#
# class PointGenerator(nn.Module):
#
#     def __init__(self):
#         super().__init__()
#         self.in_chan = IN_CHAN
#         h_dim = [256, 1024, 256, 256, 256]
#         self.m = 2048
#         self.m_training = 128
#         self.n_samples = 1
#         self.sample_dim = 8
#         self.h = h_dim[1]
#         self.map_latent = LinearBlock(Z_DIM, h_dim[0])
#         self.map_latent1 = LinearBlock(h_dim[0], h_dim[1], act=nn.ReLU())
#         self.map_latent2 = LinearBlock(h_dim[0], h_dim[1], act=None)
#         self.dbr = PointsConvBlock4(self.sample_dim, h_dim[1], act=None)
#         modules = []
#         for i in range(1, len(h_dim) - 1):
#             modules.append(PointsConvBlock4(h_dim[i], h_dim[i + 1]))
#         self.mlp = nn.Sequential(*modules)
#         self.lin = nn.Linear(h_dim[-1], IN_CHAN)
#
#     def forward(self, z):
#         batch = z.size()[0]
#         device = z.device
#         z = self.map_latent(z)
#         mul = torch.relu(self.map_latent1(z))
#         mul = mul.view(-1, 1, 1, self.h)
#         mul = mul.expand(-1, self.n_samples, -1, -1)
#         add = torch.relu(self.map_latent2(z))
#         add = add.view(-1, 1, 1, self.h)
#         add = add.expand(-1, self.n_samples, -1, -1)
#         x = torch.rand(batch, self.n_samples, self.m, self.sample_dim).to(device)
#         x = self.dbr(x, self.n_samples, self.m)
#         x = x * mul + add
#         x = self.mlp(x)
#         x = self.lin(x)
#         x = torch.tanh(x)
#         return x.squeeze()
#
#     @property
#     def m(self):
#         if self.training:
#             return self.m_training
#         else:
#             return self._m
#
#     @m.setter
#     def m(self, m):
#         self._m = m


class PointGenerator(nn.Module):

    def __init__(self):
        super().__init__()
        self.in_chan = IN_CHAN
        h_dim = [256, 1024, 128, 256, 128]
        self.m = 2048
        self.m_training = 128
        self.sample_dim = 16
        self.h = h_dim[1]
        self.map_latent1 = LinearBlock(Z_DIM, h_dim[1], act=nn.ReLU())
        self.map_latent2 = LinearBlock(Z_DIM, h_dim[1], act=nn.Sigmoid())
        self.map_latent3 = LinearBlock(Z_DIM, h_dim[1])
        self.dbr = PointsConvBlock(self.sample_dim, h_dim[1], act=None)
        modules = []
        for i in range(1, len(h_dim) - 1):
            modules.append(PointsConvBlock(h_dim[i], h_dim[i + 1]))
        modules.append(nn.Linear(h_dim[-1], IN_CHAN))
        self.mlp = nn.Sequential(*modules)

    def forward(self, z, s=None):
        batch = z.size()[0]
        device = z.device
        mul1 = self.map_latent1(z).unsqueeze(1)
        mul2 = self.map_latent2(z).unsqueeze(1)
        add = self.map_latent3(z).unsqueeze(1)
        x = s if s is not None else torch.rand(batch, self.m, self.sample_dim).to(device)
        x = self.dbr(x)
        x = x * mul1 + add * mul2
        x = self.mlp(x)
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


class FoldingNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_chan = IN_CHAN
        # Sample the grids in 2D space
        num_grid = 45
        xx = np.linspace(-0.3, 0.3, num_grid, dtype=np.float32)
        yy = np.linspace(-0.3, 0.3, num_grid, dtype=np.float32)
        self.grid = np.meshgrid(xx, yy)  # (2, 45, 45)
        self.grid = torch.Tensor(self.grid).view(2, -1)  # (2, 45, 45) -> (2, 45 * 45)
        self.fold1 = FoldingLayer(Z_DIM + 2, [512, 512, 3])
        self.fold2 = FoldingLayer(Z_DIM + 3, [512, 512, 3])

    def forward(self, z):
        batch_size = z.shape[0]

        # repeat grid for batch operation
        grid = self.grid.to(z.device)  # (2, 45 * 45)
        grid = grid.unsqueeze(0).repeat(batch_size, 1, 1)  # (B, 2, 45 * 45)

        # repeat codewords
        x = z.unsqueeze(2).repeat(1, 1, self.m)

        # two folding operations
        recon1 = self.fold1(grid, x)
        recon2 = self.fold2(recon1, x)

        return recon2


class FoldingLayer(nn.Module):
    """
    The folding operation of FoldingNet
    """

    def __init__(self, in_channel: int, out_channels: list):
        super(FoldingLayer, self).__init__()

        layers = []
        for oc in out_channels[:-1]:
            conv = nn.Conv1d(in_channel, oc, 1)
            bn = nn.BatchNorm1d(oc)
            active = nn.ReLU(inplace=True)
            layers.extend([conv, bn, active])
            in_channel = oc
        out_layer = nn.Conv1d(in_channel, out_channels[-1], 1)
        layers.append(out_layer)
        self.layers = nn.Sequential(*layers)

    def forward(self, grids, x):
        """
        Parameters
        ----------
            grids: reshaped 2D grids or intermediam reconstructed point clouds
        """
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
