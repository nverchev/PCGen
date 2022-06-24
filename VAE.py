# @title Libraries
import torch
import torch.nn as nn
from abc import ABCMeta, abstractmethod
from modules import View, DBR, MaxperChannel, PointNetEncoder, PointNetGenerator, PointGenerator
#from pointnet_modules import PCT, SPCT


class Abstract_VAE(nn.Module, metaclass=ABCMeta):
    settings = {}

    def __init__(self, in_chan, h_chan, z_dim):
        super().__init__()
        self.in_chan = in_chan
        self.h_chan = h_chan
        self.z_dim = z_dim
        self.encode = self._encoder()
        self.decode = self._decoder()
        # used for nll
        self.sigma6 = nn.Parameter(torch.tensor(0.01))

    def forward(self, x):
        data = self.encoder(x)
        return self.decoder(data)

    def sampling(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu) if self.training else mu

    def encoder(self, x):
        data = {}
        x = self.encode(x)
        if isinstance(x, list):
            data['trans'] = x[1]
            x = x[0]
        data['mu'], data['log_var'] = x.chunk(2, 1)
        data['z'] = self.sampling(data['mu'], data['log_var'])
        return data

    def decoder(self, data):
        z = data['z']
        x = self.decode(z)
        if isinstance(x, list):
            data['trans'] = x[1]
            x = x[0]
        data['recon'] = x
        return data

    @abstractmethod
    def _encoder(self):
        pass

    @abstractmethod
    def _decoder(self):
        pass

    def print_total_parameters(self):
        num_params = 0
        for param in self.parameters():
            num_params += param.numel()
        print('Total Parameters: {}'.format(num_params))
        return

    # Same architecture from https://arxiv.org/pdf/1707.02392.pdf


class Base_VAE(Abstract_VAE):

    def __init__(self, in_chan=3, h_chan=[64, 64, 64, 128, 128, 512], z_dim=128):
        self.n_points = 2048
        super().__init__(in_chan, h_chan, z_dim)

    def _encoder(self):
        modules = [DBR(self.in_chan, self.h_chan[0])]
        for i in range(len(self.h_chan) - 2):
            modules.append(DBR(self.h_chan[i], self.h_chan[i + 1]))
        modules.append(MaxperChannel())
        modules.append(DbR(self.h_chan[-2], self.h_chan[-1]))
        modules.append(nn.Linear(self.h_chan[-1], 2 * self.z_dim))

        return nn.Sequential(*modules)
    def _decoder(self):
        net = nn.Sequential(nn.Linear(self.z_dim, 256),
                            nn.ReLU(),
                            nn.Linear(256, 256),
                            nn.ReLU(),
                            nn.Linear(256, self.in_chan * self.n_points),
                            View(-1, self.n_points, self.in_chan)
                            )
        return net


class PointNet_VAE(Base_VAE):

    def __init__(self, in_chan=3, h_chan=[64, 128, 128, 256], z_dim=128):
        super(Base_VAE, self).__init__(in_chan, h_chan, z_dim)

    def _encoder(self):
        return PointNetEncoder(self.in_chan, self.h_chan, self.z_dim)


class VAE_Gen(Base_VAE):

    def __init__(self, in_chan=3, h_chan=[64, 128, 128, 256],
                 z_dim=128, m_training=2048):
        self.m_training = m_training
        super(Base_VAE, self).__init__(in_chan, h_chan, z_dim)

    def _decoder(self, m=128):
        return PointGenerator(self.in_chan, self.h_chan,
                                 self.z_dim, self.m_training)


# class PCTVAE(Base_VAE):
#
#     def __init__(self, in_chan=3, h_chan=[64, 128, 128, 256], z_dim=128):
#         super(Base_VAE, self).__init__(in_chan, h_chan, z_dim)
#
#     def _encoder(self):
#         return PCT(enc=self.z_dim)


def get_vae(model_name):
    model_dict = {
        "BaseVAE": Base_VAE,
        "PointNet": PointNet_VAE,
        "VAE_Gen": VAE_Gen,
        #'PCTVAE': PCTVAE
    }
    return model_dict[model_name]()
