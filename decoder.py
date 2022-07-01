import torch
import torch.nn as nn
import torch.nn.functional as F
from encoder import Z_DIM, IN_CHAN, N_POINTS
from modules import PointsConvBlock, LinearBlock, PointsConvBlock4, View


class MLPDecoder(nn.Module):

    def __init__(self):
        super().__init__()
        h_dim= [256, 256]
        modules = [nn.Linear(Z_DIM, h_dim[0])]
        for i in range(len(h_dim) - 1):
            modules.append(nn.ELU())
            modules.append(nn.Linear(h_dim[i], h_dim[i+1]))
        modules.append(nn.ELU())
        modules.append(nn.Linear(h_dim[-1], IN_CHAN * N_POINTS))
        modules.append(View(-1, N_POINTS, IN_CHAN))
        self.mlp = nn.Sequential(*modules)

    def forward(self, z):
        x = torch.tanh(self.mlp(z))
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
        x = torch.tanh(x)
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


class FoldingNet():
    pass


def get_decoder(decoder_name):
    decoder_dict = {
        "MLP": MLPDecoder,
        "Gen": PointGenerator,
        "FoldingNet": FoldingNet,
    }
    return decoder_dict[decoder_name]
