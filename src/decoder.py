import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from src.layer import PointsConvLayer, LinearLayer
from src.neighbour_op import graph_filtering

OUT_CHAN = 3


# class WDecoder(nn.Module):
#
#     def __init__(self, w_dim, z_dim, embedding_dim, book_size, dropout):
#         super().__init__()
#         self.w_dim = w_dim
#         self.embedding_dim = embedding_dim
#         self.book_size = book_size
#         self.dim_codes = w_dim // embedding_dim
#         self.expand = 16
#         self.h_dim = [z_dim, w_dim //2,  w_dim, self.expand * w_dim]
#         self.do = [dropout, dropout, dropout]
#         modules = [nn.BatchNorm1d(self.h_dim[0])]
#         for in_dim, out_dim, do in zip(self.h_dim, self.h_dim[1:], self.do):
#             modules.append(LinearLayer(in_dim, out_dim))
#             modules.append(nn.Dropout(do))
#         self.decode = nn.Sequential(*modules)
#         self.conv = nn.Sequential(
#             nn.Conv1d(self.expand * self.embedding_dim, self.embedding_dim, kernel_size=1))
#
#     def forward(self, x):
#         x = self.decode(x).view(-1, self.expand * self.embedding_dim, self.w_dim // self.embedding_dim)
#         x = self.conv(x).transpose(2, 1).reshape(-1, self.w_dim)
#         return x

class WDecoder(nn.Module):

    def __init__(self, w_dim, z_dim, embedding_dim, book_size, dropout):
        super().__init__()
        self.w_dim = w_dim
        self.embedding_dim = embedding_dim
        self.dim_codes = w_dim // embedding_dim
        self.book_size = book_size
        self.h_dim = [64, 64]
        self.decode = nn.Sequential(
            PointsConvLayer(self.dim_codes * z_dim, self.dim_codes * self.h_dim[0], groups=self.dim_codes),
            nn.Dropout1d(dropout),
            PointsConvLayer(self.dim_codes * self.h_dim[0], self.dim_codes * self.h_dim[1], groups=self.dim_codes),
            nn.Dropout1d(dropout),
            PointsConvLayer(self.dim_codes * self.h_dim[1], w_dim, groups=self.dim_codes,
                            batch_norm=False, act_cls=None))

    def forward(self, x):
        x = self.decode(x.repeat(1, self.dim_codes).unsqueeze(2))
        return x.squeeze(2)


class FullyConnected(nn.Module):

    def __init__(self, w_dim, m_training, filtering, **_):
        super().__init__()
        self.w_dim = w_dim
        self.h_dim = [256] * 2
        self.filtering = filtering
        self.m = m_training
        modules = [LinearLayer(w_dim, self.h_dim[0], batch_norm=False, act_cls=nn.ReLU),
                   LinearLayer(self.h_dim[0], self.h_dim[1], batch_norm=False, act_cls=nn.ReLU),
                   LinearLayer(self.h_dim[1], OUT_CHAN * self.m, batch_norm=False, act_cls=None)]
        self.mlp = nn.Sequential(*modules)

    def forward(self, w, _):  # for this model, m is fixed
        x = self.mlp(w)
        return x.view(-1, OUT_CHAN, self.m)


class FoldingBlock(nn.Module):

    def __init__(self, in_channel: int, h_dim: list, out_dim: int):
        super().__init__()
        modules = [nn.Conv1d(in_channel, h_dim[0], kernel_size=1)]
        for in_dim, out_dim in zip(h_dim, h_dim[1:] + [out_dim]):
            modules.extend([nn.ReLU(inplace=True), nn.Conv1d(in_dim, out_dim, kernel_size=1)])
        self.layers = nn.Sequential(*modules)

    def forward(self, grids, w):
        x = torch.cat([grids, w], dim=1).contiguous()
        x = self.layers(x)
        return x


class FoldingNet(nn.Module):
    def __init__(self, w_dim, m_training, filtering, **model_settings):
        super().__init__()
        self.w_dim = w_dim
        self.h_dim = [512] * 4
        self.filtering = filtering
        self.m = m_training
        self.num_grid = round(np.sqrt(m_training))
        self.m_grid = self.num_grid ** 2
        xx = torch.linspace(-0.3, 0.3, self.num_grid, dtype=torch.float)
        yy = torch.linspace(-0.3, 0.3, self.num_grid, dtype=torch.float)
        self.grid = nn.Parameter(torch.stack(torch.meshgrid(xx, yy, indexing='ij')).view(2, -1), requires_grad=False)
        self.fold1 = FoldingBlock(w_dim + 2, self.h_dim[0:2], OUT_CHAN)
        self.fold2 = FoldingBlock(w_dim + 3, self.h_dim[2:4], OUT_CHAN)
        self.graph_r = 1e-12
        self.graph_eps = 0.02
        self.graph_eps_sqr = self.graph_eps ** 2
        self.graph_lam = 0.5

    def forward(self, w, m, grid=None):  # for this model, m is fixed
        batch_size = w.shape[0]
        if grid is None:
            grid = self.grid.unsqueeze(0).repeat(batch_size, 1, 1)
        w = w.unsqueeze(2).repeat(1, 1, self.m_grid)
        x = self.fold1(grid, w)
        x = self.fold2(x, w)
        if self.filtering:
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
        delta_x = torch.BoolTensor(delta_x > self.graph_r) * delta_x
        delta_y = torch.BoolTensor(delta_y > self.graph_r) * delta_y

        delta_x_left = F.pad(delta_x, pad=[0, 0, 1, 0])
        delta_x_right = F.pad(delta_x, pad=[0, 0, 0, 1])
        delta_y_top = F.pad(delta_y, pad=[1, 0])
        delta_y_bottom = F.pad(delta_y, pad=[0, 1])

        delta = torch.stack((delta_x_left, delta_x_right, delta_y_top, delta_y_bottom), dim=1)
        delta = torch.sum(delta, dim=1).unsqueeze(1).expand(-1, 3, -1, -1)
        delta_x = delta_x.unsqueeze(1).expand(-1, 3, -1, -1)
        delta_y = delta_y.unsqueeze(1).expand(-1, 3, -1, -1)
        pc_filter1 = F.pad((pc_exp[:, :, :-1, :] * delta_x), [0, 0, 1, 0])
        pc_filter1 += F.pad((pc_exp[:, :, 1:, :] * delta_x), [0, 0, 0, 1])
        pc_filter1 += F.pad((pc_exp[:, :, :, :-1] * delta_y), [1, 0])
        pc_filter1 += F.pad((pc_exp[:, :, :, 1:] * delta_y), [0, 1])
        pc_filter = (1 - self.graph_lam * delta) * pc_exp + self.graph_lam * pc_filter1
        pc_filter = pc_filter.view(batch_size, 3, -1)
        return pc_filter


class TearingNet(FoldingNet):
    def __init__(self, w_dim, m_training, filtering, **_):
        super().__init__(w_dim, m_training, filtering=filtering)
        self.h_dim.extend([512, 512, 64, 512, 512, 2])
        modules = [nn.Conv2d(self.w_dim + 5, self.h_dim[4], kernel_size=1)]
        for in_dim, out_dim in zip(self.h_dim[4:6], self.h_dim[5:7]):
            modules.append(nn.ReLU(inplace=True))
            modules.append(nn.Conv2d(in_dim, out_dim, kernel_size=1))
        self.tearing1 = nn.Sequential(*modules)

        modules = [nn.Conv2d(self.w_dim + 5 + self.h_dim[6], self.h_dim[7], kernel_size=1)]
        for in_dim, out_dim in zip(self.h_dim[7:9], self.h_dim[8:10]):
            modules.append(nn.ReLU(inplace=True))
            modules.append(nn.Conv2d(in_dim, out_dim, kernel_size=1))
        self.tearing2 = nn.Sequential(*modules)

    def forward(self, w, m, grid=None):  # for this model, m is fixed
        batch_size = w.shape[0]
        grid = self.grid.to(w.device)
        grid = grid.unsqueeze(0).repeat(batch_size, 1, 1)
        x = super().forward(w, grid)
        grid_exp = grid.view(batch_size, 2, self.num_grid, self.num_grid)
        x_exp = x.view(-1, 3, self.num_grid, self.num_grid)
        w_exp = w.view(-1, self.w_dim, 1, 1).expand(-1, -1, self.num_grid, self.num_grid)
        in1 = torch.cat((grid_exp, x_exp, w_exp), 1).contiguous()
        # Compute the torn 2D grid
        out1 = self.tearing1(in1)  # 1st tearing
        in2 = torch.cat((in1, out1), 1)
        out2 = self.tearing2(in2)  # 2nd tearing
        out2 = out2.reshape(batch_size, 2, self.num_grid * self.num_grid)
        grid = grid + out2
        x = super().forward(w, grid)
        if self.filtering:
            x = self.graph_filter(x, grid, batch_size)
        return x


# AtlasNet
class AtlasNetv2(nn.Module):
    """AtlasNet base class"""
    patch_embed_dim = 2

    def __init__(self, w_dim, n_components, filtering, laplacian_filter, **_):
        super().__init__()
        self.w_dim = w_dim
        self.filtering = filtering
        self.laplacian = laplacian_filter
        self.num_patches = n_components if n_components else 10
        self.embedding_dim = self.w_dim + self.patch_embed_dim
        self.h_dim = [128]
        self.decoder = nn.ModuleList([self.get_mlp_adj() for _ in range(self.num_patches)])
        self.patchDeformation = [lambda x: x for _ in range(self.num_patches)]

    def get_mlp_adj(self):
        dim = self.embedding_dim
        dims = [dim, dim // 2, dim // 4]
        modules = [PointsConvLayer(dim, dim, act_cls=nn.ReLU)]
        for in_dim, out_dim in zip(dims[0:-1], dims[1:]):
            modules.append(PointsConvLayer(in_dim, out_dim, act_cls=nn.ReLU))
        modules.append(PointsConvLayer(dims[-1], OUT_CHAN, batch_norm=False, act_cls=nn.Tanh))
        return nn.Sequential(*modules)

    def forward(self, w, m):
        batch = w.size(0)
        device = w.device
        outs = []
        for i in range(self.num_patches):
            rand_grid = torch.rand(batch, 2, m // self.num_patches, device=device)
            rand_grid = self.patchDeformation[i](rand_grid)
            y = w.unsqueeze(2).expand(-1, -1, m // self.num_patches).contiguous()
            y = torch.cat((rand_grid, y), 1).contiguous()
            outs.append(self.decoder[i](y))
        x = torch.cat(outs, 2).contiguous()
        if self.filtering:
            x = graph_filtering(x, self.laplacian)
        return x


class AtlasNetv2Deformation(AtlasNetv2):
    """AtlasNet PatchDeformMLPAdj"""
    patch_embed_dim = 10

    def __init__(self, w_dim, n_components, filtering, laplacian_filter, **_):
        super().__init__(w_dim, n_components, filtering, laplacian_filter)
        self.patchDeformation = nn.ModuleList(self.get_patch_deformation() for _ in range(self.num_patches))

    def get_patch_deformation(self):
        dim = self.h_dim[0]
        modules = [PointsConvLayer(2, dim, act_cls=nn.ReLU),
                   PointsConvLayer(dim, dim, act_cls=nn.ReLU),
                   PointsConvLayer(dim, self.patch_embed_dim, batch_norm=False, act_cls=nn.Tanh)]
        return nn.Sequential(*modules)


class AtlasNetV2Translation(AtlasNetv2):
    """AtlasNet PointTranslationMLPAdj"""
    patch_embed_dim = 10

    def __init__(self, w_dim, m_training, n_components, filtering, laplacian_filter, **_):
        super().__init__(w_dim, n_components, filtering, laplacian_filter)
        self.m_patch = m_training // self.num_patches
        self.grid = nn.Parameter(torch.rand(self.num_patches, 2, m_training // self.num_patches))
        self.grid.data = F.pad(self.grid, (0, 0, 0, self.patch_embed_dim - 2))

    def forward(self, w, m):  # for this model, m is fixed
        batch = w.size(0)
        outs = []
        for i in range(self.num_patches):
            rand_grid = self.grid[i].expand(batch, -1, -1)
            y = w.unsqueeze(2).expand(-1, -1, self.m_patch).contiguous()
            y = torch.cat((rand_grid, y), 1).contiguous()
            outs.append(self.decoder[i](y))
        x = torch.cat(outs, 2).contiguous()
        if self.filtering:
            x = graph_filtering(x, self.laplacian)
        return x


class PCGenBase(nn.Module):
    concat = False

    def __init__(self, n_components, sample_dim, filtering=True, **model_settings):
        super().__init__()
        # We keep the following variable in the instantiation for compatibility with other decoders
        # w_dim ==  model_settings['hidden_dims'][1]
        # m_training given at runtime
        self.h_dim = model_settings['hidden_dims']
        self.laplacian = model_settings['laplacian_filter']
        self.act_cls = model_settings['act_cls']
        self.filtering = filtering
        self.sample_dim = sample_dim
        self.n_components = n_components
        self.map_sample1 = PointsConvLayer(self.sample_dim, self.h_dim[0], batch_norm=False, act_cls=self.act_cls)
        self.map_sample2 = PointsConvLayer(self.h_dim[0], self.h_dim[1], batch_norm=False,
                                           **{'act_cls': nn.Hardtanh} if not self.concat else {})
        self.group_conv = nn.ModuleList()
        self.group_final = nn.ModuleList()

        # PCGenConcat inherits the class and needs twice the channels here
        if self.concat:
            self.h_dim[1] = 2 * self.h_dim[1]
        for _ in range(self.n_components):
            modules = []
            for in_dim, out_dim in zip(self.h_dim[1:-1], self.h_dim[2:]):
                modules.append(PointsConvLayer(in_dim, out_dim, act_cls=self.act_cls))
            self.group_conv.append(nn.Sequential(*modules))
            self.group_final.append(PointsConvLayer(self.h_dim[-1], OUT_CHAN, batch_norm=False, act_cls=None))
        if n_components > 1:
            self.att = PointsConvLayer(self.h_dim[-1] * n_components, n_components, batch_norm=False, act_cls=None)

    def forward(self, w, m, s=None, viz_att=None, viz_components=None):
        batch = w.size()[0]
        device = w.device
        x = s if s is not None else torch.randn(batch, self.sample_dim, m, device=device)
        x = x / torch.linalg.vector_norm(x, dim=1, keepdim=True)
        x = self.map_sample1(x)
        x = self.map_sample2(x)
        x = self.join_operation(x, w)
        xs = []
        group_atts = []
        for group in range(self.n_components):
            x_group = self.group_conv[group](x)
            group_atts.append(x_group)
            x_group = self.group_final[group](x_group)
            xs.append(x_group)
        xs = torch.stack(xs, dim=3)
        if self.n_components > 1:
            x_att = F.gumbel_softmax(self.att(torch.cat(group_atts, dim=1).contiguous()), tau=8., dim=1)
            x_att = x_att.transpose(2, 1)
            x = (xs * x_att.unsqueeze(1)).sum(3)
            if viz_att is not None:  # accessory information for visualization
                assert x_att.shape == viz_att.shape, (f'Shape tensor_out {viz_att.shape} does not match shape '
                                                      f'attention {x_att.shape}')
                # side effects
                viz_att.data = x_att
            if viz_components is not None:  # accessory information for visualization
                assert xs.shape == viz_components.shape, (f'Shape tensor_out {viz_components.shape} does '
                                                          f'not match shape components {xs.shape}')
                # side effects
                viz_components.data = xs
        else:
            x = xs.squeeze(3)
        if self.filtering:
            x = graph_filtering(x, self.laplacian)
        return x

    @staticmethod
    def join_operation(x, w):
        raise NotImplementedError

    # Currently does not help inference time (also need debugging)
    # def make_parallel(self):
    #     modules = []
    #     for point_conv_list in zip(*self.group_conv):
    #         in_dim = point_conv_list[0].in_dim
    #         out_dim = point_conv_list[0].out_dim
    #         point_conv_group = PointsConvLayer(in_dim * self.num_groups, out_dim * self.num_groups, act=self.act)
    #         point_conv_group.dense=self.from_list_to_groups_conv([point_conv.dense for point_conv in point_conv_list])
    #         point_conv_group.bn = self.from_list_to_groups_bn([point_conv.bn for point_conv in point_conv_list])
    #         modules.append(point_conv_group)
    #     self.group_conv = nn.Sequential(*modules)
    #     group_conv_dense = self.from_list_to_groups_conv([point_conv.dense for point_conv in self.group_final])
    #     self.group_final = PointsConvLayer(self.h_dim[-1] * self.num_groups, OUT_CHAN * self.num_groups,
    #                                        batch_norm=False, act=None)
    #     self.group_final.dense = group_conv_dense
    #     self.parallel = True
    #
    # @staticmethod
    # def from_list_to_groups_conv(lin_list):
    #     assert all(isinstance(lin, nn.Conv1d) for lin in lin_list)
    #     assert all(lin.kernel_size == (1,) for lin in lin_list)
    #     groups = len(lin_list)
    #     concat_weights = torch.cat([lin.weight for lin in lin_list])
    #     group_lin = nn.Conv1d(concat_weights.shape[0], concat_weights.shape[1], kernel_size=1, groups=groups)
    #     group_lin.weight = nn.Parameter(concat_weights)
    #     if lin_list[0].bias is None:
    #         group_lin.bias = None
    #     else:
    #         concat_bias = torch.concat([lin.bias for lin in lin_list])
    #         group_lin.bias = nn.Parameter(concat_bias)
    #     return group_lin
    #
    # @staticmethod
    # def from_list_to_groups_bn(bn_list):
    #     assert all(isinstance(bn, nn.BatchNorm1d) for bn in bn_list)
    #     device = bn_list[0].running_mean.device
    #     concat_mean = torch.concat([bn.running_mean for bn in bn_list])
    #     concat_var = torch.concat([bn.running_var for bn in bn_list])
    #     concat_bias = torch.concat([bn.bias for bn in bn_list])
    #     group_bn = nn.BatchNorm1d(concat_mean.shape[0]).to(device)
    #     group_bn.running_mean = concat_mean.to(device)
    #     group_bn.concat_var = concat_var.to(device)
    #     group_bn.bias = nn.Parameter(concat_bias.to(device))
    #     return group_bn

    # parallel forward bit of code
    # xs = x.repeat(1, self.num_groups, 1)
    # xs = self.group_conv(xs)
    # group_att = xs
    # xs = self.group_final(xs).reshape(-1, self.num_groups, OUT_CHAN, m)
    # if self.num_groups > 1:
    #     x_att = F.gumbel_softmax(self.att(group_att), tau=8., dim=1).unsqueeze(2)
    #     x = (xs * x_att).sum(1)
    # else:
    #     x = xs


class PCGen(PCGenBase):

    @staticmethod
    def join_operation(x, w):
        return w.unsqueeze(2) * x


class PCGenConcat(PCGenBase):
    concat = True

    @staticmethod
    def join_operation(x, w):
        return torch.cat((w.unsqueeze(2).expand_as(x), x), dim=1)


def get_decoder(decoder_name):
    decoder_dict = {
        'Full': FullyConnected,
        'FoldingNet': FoldingNet,
        'TearingNet': TearingNet,
        'AtlasNetDeformation': AtlasNetv2Deformation,
        'AtlasNetTranslation': AtlasNetV2Translation,
        'PCGen': PCGen,
        'PCGenConcat': PCGenConcat,
    }
    return decoder_dict[decoder_name]
