import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from src.layer import PointsConvLayer, LinearLayer
from src.neighbour_op import graph_filtering
from pykeops.torch import LazyTensor

OUT_CHAN = 3


class CWDecoder(nn.Module):

    def __init__(self, cw_dim, z_dim):
        super().__init__()
        self.h_dim = [z_dim * 4, z_dim * 16]
        modules = [LinearLayer(z_dim, self.h_dim[0])]
        for in_dim, out_dim in zip(self.h_dim[:-1], self.h_dim[1:]):
            modules.append(LinearLayer(in_dim, out_dim))
        modules.append(LinearLayer(self.h_dim[-1], cw_dim, act=None))
        self.decode = nn.Sequential(*modules)

    def forward(self, x):
        return self.decode(x)


class FullyConnected(nn.Module):

    def __init__(self, cw_dim, m, gf):
        super().__init__()
        self.cw_dim = cw_dim
        self.h_dim = [256] * 2
        self.gf = gf
        self.m = m
        modules = [LinearLayer(cw_dim, self.h_dim[0], batch_norm=False, act=nn.ReLU(inplace=True)),
                   LinearLayer(self.h_dim[0], self.h_dim[1], batch_norm=False, act=nn.ReLU(inplace=True)),
                   LinearLayer(self.h_dim[1], OUT_CHAN * m, batch_norm=False, act=None)]
        self.mlp = nn.Sequential(*modules)

    def forward(self, z):
        x = self.mlp(z)
        return x.view(-1, OUT_CHAN, self.m)


class FoldingBlock(nn.Module):

    def __init__(self, in_channel: int, h_dim: list, out_dim: int):
        super().__init__()
        modules = [nn.Conv1d(in_channel, h_dim[0], kernel_size=1)]
        for in_dim, out_dim in zip(h_dim, h_dim[1:] + [out_dim]):
            modules.extend([nn.ReLU(inplace=True), nn.Conv1d(in_dim, out_dim, kernel_size=1)])
        self.layers = nn.Sequential(*modules)

    def forward(self, grids, x):
        x = torch.cat([grids, x], dim=1).contiguous()
        x = self.layers(x)
        return x


class FoldingNet(nn.Module):
    def __init__(self, cw_dim, m, gf):
        super().__init__()
        self.cw_dim = cw_dim
        self.h_dim = [512] * 4
        self.gf = gf
        self.m = m
        self.num_grid = round(np.sqrt(m))
        self.m_grid = self.num_grid ** 2
        xx = torch.linspace(-0.3, 0.3, self.num_grid, dtype=torch.float)
        yy = torch.linspace(-0.3, 0.3, self.num_grid, dtype=torch.float)
        self.grid = nn.Parameter(torch.stack(torch.meshgrid(xx, yy, indexing='ij')).view(2, -1), requires_grad=False)
        self.fold1 = FoldingBlock(cw_dim + 2, self.h_dim[0:2], OUT_CHAN)
        self.fold2 = FoldingBlock(cw_dim + 3, self.h_dim[2:4], OUT_CHAN)
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
    def __init__(self, cw_dim, m, gf):
        super().__init__(cw_dim, m, gf=gf)
        self.h_dim.extend([512, 512, 64, 512, 512, 2])
        modules = [nn.Conv2d(self.cw_dim + 5, self.h_dim[4], kernel_size=1)]
        for in_dim, out_dim in zip(self.h_dim[4:6], self.h_dim[5:7]):
            modules.append(nn.ReLU(inplace=True))
            modules.append(nn.Conv2d(in_dim, out_dim, kernel_size=1))
        self.tearing1 = nn.Sequential(*modules)

        modules = [nn.Conv2d(self.cw_dim + 5 + self.h_dim[6], self.h_dim[7], kernel_size=1)]
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
        z_exp = z.view(-1, self.cw_dim, 1, 1).expand(-1, -1, self.num_grid, self.num_grid)
        in1 = torch.cat((grid_exp, x_exp, z_exp), 1).contiguous()
        # Compute the torn 2D grid
        out1 = self.tearing1(in1)  # 1st tearing
        in2 = torch.cat((in1, out1), 1)
        out2 = self.tearing2(in2)  # 2nd tearing
        out2 = out2.reshape(batch_size, 2, self.num_grid * self.num_grid)
        grid = grid + out2
        x = super().forward(z, grid)
        if self.gf:
            x = self.graph_filter(x, grid, batch_size)
        return x


# AtlasNet
class AtlasNetv2(nn.Module):
    """Atlas net PatchDeformMLPAdj"""
    deformed_patch_dim = 2

    def __init__(self, cw_dim, m, gf):
        super().__init__()
        self.cw_dim = cw_dim
        self.m = m
        self.gf = gf
        self.num_patches = 16
        self.m_patch = self.m // self.num_patches
        self.dim_embedding = self.cw_dim + self.deformed_patch_dim
        self.h_dim = [128]
        self.decoder = nn.ModuleList([self.get_mlp_adj() for _ in range(self.num_patches)])

    def get_mlp_adj(self):
        dim = self.dim_embedding
        dims = [dim, dim // 2, dim // 4]
        modules = [PointsConvLayer(dim, dim, act=nn.ReLU(inplace=True))]
        for in_dim, out_dim in zip(dims[0:-1], dims[1:]):
            modules.append(PointsConvLayer(in_dim, out_dim, act=nn.ReLU(inplace=True)))
        modules.append(PointsConvLayer(dims[-1], OUT_CHAN, batch_norm=False, act=nn.Tanh()))
        return nn.Sequential(*modules)

    def forward(self, x):
        batch = x.size(0)
        device = x.device
        outs = []
        for i in range(self.num_patches):
            rand_grid = torch.rand(batch, 2, self.m_patch, device=device)
            rand_grid = self.patchDeformation[i](rand_grid)
            y = x.unsqueeze(2).expand(-1, -1, self.m_patch).contiguous()
            y = torch.cat((rand_grid, y), 1).contiguous()
            outs.append(self.decoder[i](y))
        x = torch.cat(outs, 2).contiguous()
        if self.gf:
            x = graph_filtering(x)
        return x


class AtlasNetv2Deformation(AtlasNetv2):
    """Atlas net PatchDeformMLPAdj"""
    deformed_patch_dim = 10

    def __init__(self, cw_dim, m, gf):
        super().__init__(cw_dim, m, gf)
        self.patchDeformation = nn.ModuleList(self.get_patch_deformation() for _ in range(self.num_patches))

    def get_patch_deformation(self):
        dim = self.h_dim[0]
        modules = [PointsConvLayer(2, dim, act=nn.ReLU(inplace=True)),
                   PointsConvLayer(dim, dim, act=nn.ReLU(inplace=True)),
                   PointsConvLayer(dim, self.deformed_patch_dim, batch_norm=False, act=nn.Tanh())]
        return nn.Sequential(*modules)

    def forward(self, x):
        batch = x.size(0)
        device = x.device
        outs = []
        for i in range(self.num_patches):
            rand_grid = torch.rand(batch, 2, self.m_patch, device=device)
            rand_grid = self.patchDeformation[i](rand_grid)
            y = x.unsqueeze(2).expand(-1, -1, self.m_patch).contiguous()
            y = torch.cat((rand_grid, y), 1).contiguous()
            outs.append(self.decoder[i](y))
        x = torch.cat(outs, 2).contiguous()
        if self.gf:
            x = graph_filtering(x)
        return x


class AtlasNetv2Structures(AtlasNetv2):
    """Atlas net PatchDeformMLPAdj"""
    deformed_patch_dim = 10

    def __init__(self, cw_dim, m, gf):
        super().__init__(cw_dim, m, gf)
        self.grid = nn.Parameter(torch.rand(self.num_patches, self.deformed_patch_dim, self.m_patch))

    def forward(self, x):
        batch = x.size(0)
        outs = []
        for i in range(self.num_patches):
            rand_grid = self.grid[i].expand(batch, -1, -1)
            y = x.unsqueeze(2).expand(-1, -1, self.m_patch).contiguous()
            y = torch.cat((rand_grid, y), 1).contiguous()
            outs.append(self.decoder[i](y))
        x = torch.cat(outs, 2).contiguous()
        if self.gf:
            x = graph_filtering(x)
        return x


# AtlasNet with grouped convolutions. Inference time is halved by 2 but efficient backward pass has
# not yet been implemented by in pytorch.

# class AtlasNetv2(nn.Module):
#     """Atlas net PatchDeformMLPAdj"""
#
#     def __init__(self, cw_dim, m, gf):
#         super().__init__()
#         self.cw_dim = cw_dim
#         self.m = m
#         self.gf = gf
#         self.num_patch = 8
#         self.m_patch = self.m // self.num_patch
#         self.deform_patch_dim = 8
#         self.h_dim = [128]
#
#         total_in = 2 * self.num_patch
#         dim = self.h_dim[0] * self.num_patch
#         total_out = self.deform_patch_dim * self.num_patch
#         modules = [PointsConvBlock(total_in, dim, act=nn.ReLU(inplace=True), groups=self.num_patch),
#                    PointsConvBlock(dim, dim, act=nn.ReLU(inplace=True), groups=self.num_patch),
#                    PointsConvBlock(dim, total_out, batch_norm=False, act=nn.Tanh(), groups=self.num_patch)]
#         self.patchDeformation = nn.Sequential(*modules)
#
#         dim = (self.deform_patch_dim + self.cw_dim) * self.num_patch
#         dims = [dim, dim // 2, dim // 4]
#         modules = [PointsConvBlock(dim, dim, act=nn.ReLU(inplace=True), groups=self.num_patch)]
#         for in_dim, out_dim in zip(dims[0:-1], dims[1:]):
#             modules.append(PointsConvBlock(in_dim, out_dim, act=nn.ReLU(inplace=True), groups=self.num_patch))
#         modules.append(PointsConvBlock(dims[-1], OUT_CHAN * self.num_patch, batch_norm=False, act=nn.Tanh(), groups=self.num_patch))
#         self.mlp_adj = nn.Sequential(*modules)
#
#     def forward(self, z):
#         batch = z.size(0)
#         device = z.device
#         x = z.view(batch, 1, self.cw_dim, 1).expand(-1, self.num_patch, -1, self.m_patch)
#         rand_grid = torch.rand(batch, self.num_patch * 2, self.m_patch, device=device)
#         deformed_grid = self.patchDeformation(rand_grid).view(-1, self.num_patch, self.deform_patch_dim, self.m_patch)
#         x = torch.cat([deformed_grid, x], dim=2).contiguous().view(batch, -1, self.m_patch)
#         x = self.mlp_adj(x).view(batch, OUT_CHAN, self.m)
#         if self.gf:
#             x = graph_filtering(x)
#         return x


class PCGen(nn.Module):

    def __init__(self, cw_dim, m, gf=True):
        super().__init__()
        self.h_dim = [256, cw_dim, 512, 512, 512, 64]
        self.m = m
        self.m_training = m
        self.gf = gf
        self.sample_dim = 16
        self.map_sample1 = PointsConvLayer(self.sample_dim, self.h_dim[0], batch_norm=False, act=nn.ReLU(inplace=True))
        self.map_sample2 = PointsConvLayer(self.h_dim[0], self.h_dim[1], batch_norm=False,
                                           act=nn.Hardtanh(inplace=True))
        modules = []
        for in_dim, out_dim in zip(self.h_dim[1:-1], self.h_dim[2:]):
            modules.append(PointsConvLayer(in_dim, out_dim, act=nn.ReLU(inplace=True)))

        modules.append(PointsConvLayer(self.h_dim[-1], OUT_CHAN, batch_norm=False, act=None))
        self.points_convs = nn.Sequential(*modules)

    def forward(self, z, s=None):
        batch = z.size()[0]
        device = z.device
        m = self.m_training if self.training else self.m
        x = s if s is not None else torch.randn(batch, self.sample_dim, m, device=device)
        x = x / torch.linalg.vector_norm(x, dim=1, keepdim=True)
        x = self.map_sample1(x)
        x = self.map_sample2(x)
        x = z.unsqueeze(2) * x
        x = self.points_convs(x)
        if self.gf:
            x = graph_filtering(x)
        return x


# class PCGen(nn.Module):
#
#     def __init__(self, cw_dim, m, gf=True):
#         super().__init__()
#         self.h_dim = [256, cw_dim, 512, 256, 128, 64]
#         self.m = m
#         self.m_training = m
#         self.gf = gf
#         self.sample_dim = 16
#         self.map_sample1 = PointsConvLayer(self.sample_dim, self.h_dim[0], batch_norm=False, act=nn.ReLU(inplace=True))
#         self.map_sample2 = PointsConvLayer(self.h_dim[0], self.h_dim[1], batch_norm=False,
#                                            act=nn.Hardtanh(inplace=True))
#         modules = []
#         for in_dim, out_dim in zip(self.h_dim[1:-1], self.h_dim[2:]):
#             modules.append(PointsConvLayer(in_dim, out_dim, act=nn.ReLU(inplace=True)))
#
#         modules.append(PointsConvLayer(self.h_dim[-1], OUT_CHAN, batch_norm=False, act=None))
#         self.points_convs = nn.Sequential(*modules)
#
#     def forward(self, z, s=None):
#         batch = z.size()[0]
#         device = z.device
#         m = self.m_training if self.training else self.m
#         x = s if s is not None else torch.randn(batch, self.sample_dim, m, device=device)
#         x = x / torch.linalg.vector_norm(x, dim=1, keepdim=True)
#         x = self.map_sample1(x)
#         x = self.map_sample2(x)
#         x = z.unsqueeze(2) * x
#         x = self.points_convs(x)
#         if self.gf:
#             x = graph_filtering(x)
#         return x

class PCGenH(nn.Module):

    def __init__(self, cw_dim, m, gf=True):
        super().__init__()

        self.m = m
        self.m_training = m
        self.gf = gf
        self.sample_dim = 16
        h_dim = [128, cw_dim, 512, 64]
        self.h_dim = h_dim
        self.map_sample1 = PointsConvLayer(self.sample_dim, h_dim[0], batch_norm=False, act=nn.ReLU(inplace=True))
        self.map_sample2 = PointsConvLayer(h_dim[0], cw_dim, batch_norm=False,
                                           act=nn.Hardtanh(inplace=True))
        modules = []
        for in_dim, out_dim in zip(h_dim[1:-1], h_dim[2:]):
            modules.append(PointsConvLayer(in_dim, out_dim, act=nn.ReLU(inplace=True)))

        modules.append(PointsConvLayer(h_dim[-1], OUT_CHAN, batch_norm=False, act=None))
        self.points_convs1 = nn.Sequential(*modules)
        h_dim = [128, cw_dim + 3, 512, 64]
        self.h_dim.extend(h_dim)

        self.map_sample3 = PointsConvLayer(self.sample_dim, h_dim[0], batch_norm=False, act=nn.ReLU(inplace=True))
        self.map_sample4 = PointsConvLayer(h_dim[0], cw_dim, batch_norm=False,
                                           act=nn.Hardtanh(inplace=True))
        modules = []
        for in_dim, out_dim in zip(h_dim[1:-1], h_dim[2:]):
            modules.append(PointsConvLayer(in_dim, out_dim, act=nn.ReLU(inplace=True)))

        modules.append(PointsConvLayer(h_dim[-1], OUT_CHAN, batch_norm=False, act=None))
        self.points_convs2 = nn.Sequential(*modules)
        h_dim = [128, cw_dim + 3, 512, 512, 512, 64]
        self.h_dim.extend(h_dim)

        self.map_sample5 = PointsConvLayer(self.sample_dim, h_dim[0], batch_norm=False, act=nn.ReLU(inplace=True))
        self.map_sample6 = PointsConvLayer(h_dim[0], cw_dim, batch_norm=False,
                                           act=nn.Hardtanh(inplace=True))

        modules = []
        for in_dim, out_dim in zip(h_dim[1:-1], h_dim[2:]):
            modules.append(PointsConvLayer(in_dim, out_dim, act=nn.ReLU(inplace=True)))

        modules.append(PointsConvLayer(h_dim[-1], OUT_CHAN, batch_norm=False, act=None))
        self.points_convs3 = nn.Sequential(*modules)

    def forward(self, z, s=None):
        batch, cw_dim = z.size()
        device = z.device
        z = z.unsqueeze(2)
        m = self.m_training if self.training else self.m
        m_top = m // 16
        assert m_top, "Hierarchical version needs at least 16 points"
        s = s if s is not None else torch.randn(batch, self.sample_dim, m_top * 21, device=device)
        s = s / torch.linalg.vector_norm(s, dim=1, keepdim=True)
        x = s[..., :m_top]
        x = self.map_sample1(x)
        x = self.map_sample2(x)
        x = z * x
        x = self.points_convs1(x)
        x1 = x.repeat(1, 1, 4)
        x = s[..., m_top:5 * m_top]
        x = self.map_sample3(x)
        x = self.map_sample4(x)
        x = z * x
        x = torch.cat([x1, x], dim=1).contiguous()
        x = self.points_convs2(x)
        x2 = x.repeat(1, 1, 4)
        x = s[..., 5 * m_top:]
        x = self.map_sample5(x)
        x = self.map_sample6(x)
        x = z * x
        x = torch.cat([x2, x], dim=1).contiguous()
        x = self.points_convs3(x)
        if self.gf:
            x = graph_filtering(x)
        return x


class PCGenH(nn.Module):

    def __init__(self, cw_dim, m, gf=True):
        super().__init__()

        self.m = m
        self.m_training = m
        self.gf = gf
        self.sample_dim = 16
        h_dim = [256, cw_dim, 512]
        self.h_dim = h_dim
        self.map_sample1 = PointsConvLayer(self.sample_dim, h_dim[0], batch_norm=False, act=nn.ReLU(inplace=True))
        self.map_sample2 = PointsConvLayer(h_dim[0], cw_dim, batch_norm=False,
                                           act=nn.Hardtanh(inplace=True))
        modules = []
        for in_dim, out_dim in zip(h_dim[1:-1], h_dim[2:]):
            modules.append(PointsConvLayer(in_dim, out_dim, act=nn.ReLU(inplace=True)))
        self.points_convs1 = nn.Sequential(*modules)
        h_dim = [256, cw_dim, 512, 512]
        self.h_dim.extend(h_dim)

        self.map_sample3 = PointsConvLayer(self.sample_dim, h_dim[0], batch_norm=False, act=nn.ReLU(inplace=True))
        self.map_sample4 = PointsConvLayer(h_dim[0], cw_dim, batch_norm=False,
                                           act=nn.Hardtanh(inplace=True))
        modules = []
        for in_dim, out_dim in zip(h_dim[1:-1], h_dim[2:]):
            modules.append(PointsConvLayer(in_dim, out_dim, act=nn.ReLU(inplace=True)))
        self.points_convs2 = nn.Sequential(*modules)
        self.att0 = PointsConvLayer(512, 64, batch_norm=False, act=None)
        self.att1 = PointsConvLayer(512, 64, batch_norm=False, act=None)
        self.att2 = PointsConvLayer(512, 64, batch_norm=False, act=None)
        self.att3 = PointsConvLayer(512, 64, batch_norm=False, act=None)

        self.final = nn.Sequential(PointsConvLayer(64, OUT_CHAN, batch_norm=False, act=None))

    def forward(self, z, s=None):
        batch, cw_dim = z.size()
        device = z.device
        z = z.unsqueeze(2)
        m = self.m_training if self.training else self.m
        m_top = m // 16
        assert m_top, "Hierarchical version needs at least 16 points"
        s = s if s is not None else torch.randn(batch, self.sample_dim, m_top * 17, device=device)
        s = s / torch.linalg.vector_norm(s, dim=1, keepdim=True)
        x = s[..., :m_top]
        x = self.map_sample1(x)
        x = self.map_sample2(x)
        x = z * x
        x1 = self.points_convs1(x)
        x = s[..., m_top:]
        x = self.map_sample3(x)
        x = self.map_sample4(x)
        x = z * x
        x = self.points_convs2(x)
        queries = self.att0(x1)
        keys = self.att1(x1)
        values = self.att2(x)
        A = torch.softmax(torch.bmm(queries, keys.transpose(2, 1)) / np.sqrt(64), dim=1)
        x = self.att3(x) + torch.bmm(A, values)
        x = self.final(x)
        if self.gf:
            x = graph_filtering(x)
        return x


from src.neighbour_op import get_graph_features


class PCGenH(nn.Module):

    def __init__(self, cw_dim, m, gf=True):
        super().__init__()

        self.m = m
        self.m_training = m
        self.gf = gf
        self.sample_dim = 16
        h_dim = [128, cw_dim, 128]
        self.h_dim = h_dim
        self.map_sample1 = PointsConvLayer(self.sample_dim, h_dim[0], batch_norm=False, act=nn.ReLU(inplace=True))
        self.map_sample2 = PointsConvLayer(h_dim[0], cw_dim, batch_norm=False,
                                           act=nn.Hardtanh(inplace=True))
        modules = []
        for in_dim, out_dim in zip(h_dim[1:-1], h_dim[2:]):
            modules.append(PointsConvLayer(in_dim, out_dim, act=nn.ReLU(inplace=True)))

        self.points_convs1 = nn.Sequential(*modules)
        h_dim = [128, cw_dim + 256, 512, 64]
        self.h_dim.extend(h_dim)

        self.map_sample3 = PointsConvLayer(self.sample_dim, h_dim[0], batch_norm=False, act=nn.ReLU(inplace=True))
        self.map_sample4 = PointsConvLayer(h_dim[0], cw_dim, batch_norm=False,
                                           act=nn.Hardtanh(inplace=True))
        modules = []
        for in_dim, out_dim in zip(h_dim[1:-1], h_dim[2:]):
            modules.append(PointsConvLayer(in_dim, out_dim, act=nn.ReLU(inplace=True)))

        modules.append(PointsConvLayer(h_dim[-1], OUT_CHAN, batch_norm=False, act=None))
        self.points_convs2 = nn.Sequential(*modules)
        self.h_dim.extend(h_dim)

    def forward(self, z, s=None):
        batch, cw_dim = z.size()
        device = z.device
        z = z.unsqueeze(2)
        m = self.m_training if self.training else self.m
        m_top = m // 16
        assert m_top, "Hierarchical version needs at least 16 points"
        s = s if s is not None else torch.randn(batch, self.sample_dim, m_top * 17, device=device)
        s = s / torch.linalg.vector_norm(s, dim=1, keepdim=True)
        x = s[..., :m_top]
        x = self.map_sample1(x)
        x = self.map_sample2(x)
        x = z * x
        x = self.points_convs1(x)
        x1 = get_graph_features(x, 16).flatten(2)
        x = s[..., m_top:]
        x = self.map_sample3(x)
        x = self.map_sample4(x)
        x = z * x
        x = torch.cat([x1, x], dim=1).contiguous()
        x = self.points_convs2(x)
        if self.gf:
            x = graph_filtering(x)
        return x


class PCGenH(nn.Module):

    def __init__(self, cw_dim, m, gf=True):
        super().__init__()
        self.h_dim = [256, cw_dim, 512, 512, 512, 64]
        self.m = m
        self.m_training = m
        self.gf = gf
        self.sample_dim = 16
        self.num_groups = 8
        self.map_sample1 = PointsConvLayer(self.sample_dim, self.h_dim[0], batch_norm=False, act=nn.ReLU(inplace=True))
        self.map_sample2 = PointsConvLayer(self.h_dim[0], self.h_dim[1], batch_norm=False,
                                           act=nn.Hardtanh(inplace=True))
        self.group_conv = nn.ModuleList()
        self.group_final = nn.ModuleList()
        self.group_att1 = nn.ModuleList()
        self.group_att2 = nn.ModuleList()
        self.group_att3 = nn.ModuleList()
        self.group_att4 = nn.ModuleList()

        for _ in range(self.num_groups):
            modules = []
            for in_dim, out_dim in zip(self.h_dim[1:-1], self.h_dim[2:]):
                modules.append(PointsConvLayer(in_dim, out_dim, act=nn.ReLU(inplace=True)))
            self.group_conv.append(nn.Sequential(*modules))
            self.group_att1.append(PointsConvLayer(self.h_dim[-1], self.h_dim[-1], batch_norm=False, act=None))
            self.group_att2.append(PointsConvLayer(self.h_dim[-1], self.h_dim[-1], batch_norm=False, act=None))
            self.group_att3.append(PointsConvLayer(self.h_dim[-1], self.h_dim[-1], batch_norm=False, act=None))
            self.group_att4.append(PointsConvLayer(self.h_dim[-1], self.h_dim[-1], batch_norm=False, act=None))
            self.group_final.append(PointsConvLayer(self.h_dim[-1], 3, batch_norm=False, act=None))


    def forward(self, z, s=None):
        batch = z.size()[0]
        device = z.device
        m = self.m_training if self.training else self.m
        x = s if s is not None else torch.randn(batch, self.sample_dim, m, device=device)
        x = x / torch.linalg.vector_norm(x, dim=1, keepdim=True)
        x = self.map_sample1(x)
        x = self.map_sample2(x)
        group_size = m // self.num_groups
        assert group_size, f"Number of generared point should be larger than {self.num_groups}"
        x = z.unsqueeze(2) * x
        xs = []
        x_old = None
        for group in range(self.num_groups):
            x_group = x[..., group * group_size: (group + 1) * group_size]
            x_group = self.group_conv[group](x_group)
            if x_old is not None:
                keys = self.group_att1[group](x_old)
                queries = self.group_att2[group](x_old)
                values = self.group_att3[group](x_group)
                A = torch.softmax(torch.bmm(queries, keys.transpose(2, 1)), dim=2)
                x_group = x_group + self.group_att4[group](torch.bmm(A, values))

            x_old = x_group
            x_group = self.group_final[group](x_group)
            xs.append(x_group)
        x = torch.cat(xs, dim=2)
        if self.gf:
            x = graph_filtering(x)
        return x


class PCGenH(nn.Module):

    def __init__(self, cw_dim, m, gf=True):
        super().__init__()
        self.h_dim = [256, cw_dim, 512, 512, 512, 64]
        self.m = m
        self.m_training = m
        self.gf = gf
        self.sample_dim = 16
        self.num_groups = 8
        self.map_sample1 = PointsConvLayer(self.sample_dim, self.h_dim[0], batch_norm=False, act=nn.ReLU(inplace=True))
        self.map_sample2 = PointsConvLayer(self.h_dim[0], self.h_dim[1], batch_norm=False,
                                           act=nn.Hardtanh(inplace=True))
        self.group_conv = nn.ModuleList()
        self.group_final = nn.ModuleList()
        self.group_att = nn.ModuleList()

        for _ in range(self.num_groups):
            modules = []
            for in_dim, out_dim in zip(self.h_dim[1:-1], self.h_dim[2:]):
                modules.append(PointsConvLayer(in_dim, out_dim, act=nn.ReLU(inplace=True)))
            self.group_conv.append(nn.Sequential(*modules))
            self.group_final.append(PointsConvLayer(self.h_dim[-1], OUT_CHAN, act=None))
            self.group_att.append(PointsConvLayer(self.h_dim[-1], 1, act=None))

    def forward(self, z, s=None):
        batch = z.size()[0]
        device = z.device
        m = self.m_training if self.training else self.m
        x = s if s is not None else torch.randn(batch, self.sample_dim, m, device=device)
        x = x / torch.linalg.vector_norm(x, dim=1, keepdim=True)
        x = self.map_sample1(x)
        x = self.map_sample2(x)
        x = z.unsqueeze(2) * x
        xs = []
        x_atts = []
        for group in range(self.num_groups):
            x_group = self.group_conv[group](x)
            x_att = self.group_att[group](x_group)
            x_atts.append(x_att)
            x_group = self.group_final[group](x_group)
            xs.append(x_group)
        att = torch.softmax(torch.cat(x_atts, dim=1).contiguous().transpose(2, 1), dim=2)
        x = (torch.stack(xs, dim=3).contiguous() * att.unsqueeze(1)).sum(3)
        if self.gf:
            x = graph_filtering(x)
        return x


class PCGenH(nn.Module):

    def __init__(self, cw_dim, m, gf=True):
        super().__init__()
        self.h_dim = [256, cw_dim, 512, 512, 512, 64]
        self.m = m
        self.m_training = m
        self.gf = gf
        self.sample_dim = 16
        self.num_groups = 4
        self.map_sample1 = PointsConvLayer(self.sample_dim, self.h_dim[0], batch_norm=False, act=nn.ReLU(inplace=True))
        self.map_sample2 = PointsConvLayer(self.h_dim[0], self.h_dim[1], batch_norm=False, act=nn.Hardtanh(inplace=True))
        self.group_conv = nn.ModuleList()
        self.group_final = nn.ModuleList()
        for _ in range(self.num_groups):
            modules = []
            for in_dim, out_dim in zip(self.h_dim[1:-1], self.h_dim[2:]):
                modules.append(PointsConvLayer(in_dim, out_dim, act=nn.ReLU(inplace=True)))
            self.group_conv.append(nn.Sequential(*modules))
            self.group_final.append(PointsConvLayer(self.h_dim[-1], OUT_CHAN, batch_norm=False, act=None))
        self.att1 = PointsConvLayer(self.h_dim[-1] * self.num_groups,  self.num_groups, act=None)
        #self.att2 = LinearLayer(self.h_dim[-1] * self.num_groups, self.h_dim[-1] * self.num_groups, act=None)

    def forward(self, z, s=None):
        batch = z.size()[0]
        device = z.device
        m = self.m_training if self.training else self.m
        x = s if s is not None else torch.randn(batch, self.sample_dim, m, device=device)
        x = x / torch.linalg.vector_norm(x, dim=1, keepdim=True)
        x = self.map_sample1(x)
        x = self.map_sample2(x)
        x = z.unsqueeze(2) * x
        xs = []
        x_atts = []
        for group in range(self.num_groups):
            x_group = self.group_conv[group](x)
            x_atts.append(x_group)
            x_group = self.group_final[group](x_group)
            xs.append(x_group)
        x_att = torch.cat(x_atts, dim=1).contiguous()
        #keys = self.att2(x_att.max(2)[0])
        #x_att = x_att * torch.sigmoid(keys).unsqueeze(2)
        att = torch.softmax(self.att1(x_att), dim=1).transpose(2, 1)
        x = (torch.stack(xs, dim=3) * att.unsqueeze(1)).sum(3)

        if self.gf:
            x = graph_filtering(x)
        return x

def get_decoder(decoder_name):
    decoder_dict = {
        'Full': FullyConnected,
        'FoldingNet': FoldingNet,
        'TearingNet': TearingNet,
        'AtlasNetDeformation': AtlasNetv2Deformation,
        'AtlasNetStructures': AtlasNetv2Structures,
        'PCGen': PCGen,
        'PCGenH': PCGenH,

    }
    return decoder_dict[decoder_name]
