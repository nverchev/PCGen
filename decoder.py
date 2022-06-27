import torch
import torch.nn as nn
from encoder import Z_DIM, IN_CHAN, N_POINTS
from modules import DBR, DbR, DBR4, View

def get_mlp_decoder():
    net = nn.Sequential(nn.Linear(Z_DIM, 256),
                        nn.ReLU(),
                        nn.Linear(256, 256),
                        nn.ReLU(),
                        nn.Linear(256, IN_CHAN * N_POINTS),
                        View(-1, N_POINTS, IN_CHAN)
                        )
    return net


class PointGenerator(nn.Module):

    def __init__(self):
        super().__init__()
        self.in_chan = IN_CHAN
        self.m = 2048
        self.m_training = 128
        self.n_samples = 1
        self.sample_dim = 8
        self.hz = 256
        self.h = 1024
        self.h2 = 256
        self.dbr = DBR4(self.sample_dim, self.h)
        self.map_latent = DbR(Z_DIM, self.hz)
        self.map_latent1 = nn.Linear(self.hz, self.h)
        self.dbr1 = DBR4(self.h, self.h2)
        self.dbr2 = DBR4(self.h2, self.h2)
        self.dbr3 = DBR4(self.h2, self.h2)
        self.lin = nn.Linear(self.h2, IN_CHAN)

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

class FoldingNet():
    pass

def get_decoder(decoder_name):
    decoder_dict = {
        "MLP": get_mlp_decoder,
        "Gen": PointGenerator,
        "FoldingNet": FoldingNet,
    }
    return decoder_dict[decoder_name]
