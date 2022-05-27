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
