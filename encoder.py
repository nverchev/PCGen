import torch
import torch.nn as nn
from modules import DBR, DbR, STN, MaxChannel, get_conv2d
from utils import get_graph_features
from pointnet_modules import PCT, SPCT
# input feature dimension
IN_CHAN = 3
N_POINTS = 2048
Z_DIM = 128



class DGCNN(nn.Module):
    def __init__(self, feat_dim=1024,  k=20):
        super().__init__()
        self.k = k
        h_dim = [64, 64, 64, 128, 128, 512]
        self.conv = get_conv2d(2 * IN_CHAN, h_dim[0])
        modules = []
        for i in range(1, len(h_dim) - 2):
            modules.append(DBR(h_dim[i], h_dim[i + 1]))
        modules.append(MaxChannel())
        modules.append(DbR(h_dim[-2], h_dim[-1]))
        modules.append(nn.Linear(h_dim[-1], 2 * Z_DIM))
        self.encode = nn.Sequential(*modules)

    def forward(self, x):
        x = x.transpose(2, 1).contiguous()
        x = get_graph_features(x, k=self.k)
        x = self.conv(x)
        x = x.max(dim=-1, keepdim=False)[0]
        x = x.transpose(2, 1).contiguous()
        return self.encode(x)



def get_mlp_encoder():
    h_dim = [64, 64, 64, 128, 128, 512]
    modules = [DBR(IN_CHAN, h_dim[0])]
    for i in range(len(h_dim) - 2):
        modules.append(DBR(h_dim[i], h_dim[i + 1]))
    modules.append(MaxChannel())
    modules.append(DbR(h_dim[-2], h_dim[-1]))
    modules.append(nn.Linear(h_dim[-1], 2 * Z_DIM))
    return nn.Sequential(*modules)


class PointNetEncoder(nn.Module):

    def __init__(self):
        super().__init__()
        h_chan = [64, 64, 64, 128, 128, 512]
        self.stn = STN()
        self.stnk = STN(h_chan[0])
        self.dbr1 = DBR(IN_CHAN, h_chan[0])
        modules = []
        for i in range(len(h_chan) - 2):
            modules.append(DBR(h_chan[i], h_chan[i + 1]))
        modules.append(MaxChannel())
        modules.append(DbR(h_chan[-2], h_chan[-1]))
        modules.append(nn.Linear(h_chan[-1], 2 * Z_DIM))
        self.encode = nn.Sequential(*modules)

    def forward(self, x):
        trans = self.stn(x)
        x = torch.matmul(x, trans)
        x = self.dbr1(x)
        trans = self.stnk(x)
        x = torch.matmul(x, trans)
        x = self.encode(x)
        return [x, trans]



def get_encoder(encoder_name):
    dict_encoder = {
        "MLP": get_mlp_encoder,
        "PointNet": PointNetEncoder,
        "DGCNN": DGCNN,
        "PCT": PCT
    }
    return dict_encoder[encoder_name]