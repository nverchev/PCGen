"""Point cloud decoders and reconstruction heads."""

import abc
import itertools
from typing import Final, NamedTuple, override

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config.experiment import Experiment
from src.config.options import Decoders
from src.data.structures import OUT_CHAN
from src.module.layers import (
    ActClass,
    AppendCodeword,
    GraphFilter,
    LinearLayer,
    Modulate,
    PointsConvLayer,
    PrependCodeword,
    Repeat,
    SphereProjection,
)

__all__ = [
    'AtlasNetV2Translation',
    'AtlasNetv2',
    'AtlasNetv2Deformation',
    'BasePointDecoder',
    'BaseWDecoder',
    'ComponentAttention',
    'Decoded',
    'FoldingBlock',
    'FoldingNet',
    'FullyConnected',
    'GridFilter',
    'PCGen',
    'PCGenBase',
    'PCGenConcat',
    'TearingNet',
    'WDecoder',
    'get_decoder',
]


class Decoded(NamedTuple):
    """What a decoder produces.

    Attributes:
        cloud: Decoded coordinates ``(B, 3, M)``.
        attention: Share of every point each component claims ``(B, M, C)``,
            empty unless the decoder mixes several components.
        components: Coordinates every component decodes ``(B, 3, M, C)``,
            empty unless the decoder mixes several components.
    """

    cloud: torch.Tensor
    attention: torch.Tensor = torch.empty(0)
    components: torch.Tensor = torch.empty(0)


class BaseWDecoder(nn.Module, abc.ABC):
    """Abstract base class for codeword decoders."""

    w_dim: Final[int]
    embedding_dim: Final[int]
    dim_codes: Final[int]
    book_size: Final[int]
    act_cls: Final[ActClass]
    bn_momentum: Final[float]
    z_dim: Final[int]
    dropout: Final[float]
    h_dims: list[int]
    h_dim: list[int]

    def __init__(self) -> None:
        """Initialize."""
        super().__init__()
        cfg = Experiment.get_config()
        self.w_dim = cfg.autoencoding.model.w_dim
        self.embedding_dim = cfg.autoencoding.model.embedding_dim
        self.dim_codes = cfg.autoencoding.model.dim_codes
        self.book_size = cfg.autoencoding.model.book_size
        self.act_cls = cfg.autoencoding.architecture.act_cls
        self.bn_momentum = cfg.autoencoding.architecture.bn_momentum
        cfg_w = cfg.w_autoencoding
        assert cfg_w is not None
        self.z_dim = cfg_w.model.z_dim
        self.dropout = cfg_w.architecture.dropout
        self.h_dims = list(cfg_w.architecture.decoder_h_dims)
        self.h_dim = self.h_dims
        return

    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Decode latent vector to codeword."""
        ...


class WDecoder(BaseWDecoder):
    """Decoder mapping continuous latent z back to codeword representation."""

    layers: nn.Sequential

    def __init__(self) -> None:
        """Initialize."""
        super().__init__()
        self.layers = nn.Sequential(
            Repeat(1, self.dim_codes),
            nn.Unflatten(1, (self.dim_codes * self.z_dim, 1)),
            *self._get_conv_layers(),
            PointsConvLayer(
                self.dim_codes * self.h_dims[-1],
                self.w_dim,
                groups=self.dim_codes,
                batch_norm=False,
                act_cls=None,
            ),
            nn.Flatten(),
        )
        return

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Decode latent vector to codeword."""
        return self.layers(x)

    def _get_conv_layers(self) -> list[nn.Module]:
        conv_layers: list[nn.Module] = []
        for in_d, out_d in itertools.pairwise((self.z_dim, *self.h_dims)):
            conv_layers.extend(
                [
                    PointsConvLayer(
                        self.dim_codes * in_d,
                        self.dim_codes * out_d,
                        groups=self.dim_codes,
                        act_cls=self.act_cls,
                        bn_momentum=self.bn_momentum,
                    ),
                    nn.Dropout1d(self.dropout),
                ]
            )

        return conv_layers


class BasePointDecoder(nn.Module, abc.ABC):
    """Abstract base class for point cloud decoders."""

    w_dim: Final[int]
    act_cls: Final[ActClass]
    bn_momentum: Final[float]
    filtering: Final[bool]
    n_components: Final[int]
    m_training: Final[int]
    h_dim: list[int]

    def __init__(self) -> None:
        """Initialize."""
        super().__init__()
        cfg = Experiment.get_config().autoencoding
        self.w_dim = cfg.model.w_dim
        self.act_cls = cfg.architecture.act_cls
        self.bn_momentum = cfg.architecture.bn_momentum
        self.filtering = cfg.architecture.decoder.filtering
        self.n_components = cfg.architecture.decoder.n_components
        self.m_training = cfg.train.m_training
        return

    @abc.abstractmethod
    def forward(
        self,
        w: torch.Tensor,
        m: int,
        s: torch.Tensor,
    ) -> Decoded:
        """Decode codeword ``(B, W)`` into point cloud coordinates ``(B, 3, M)``."""
        ...


class GridFilter(nn.Module):
    """Smooth a cloud folded from a grid, along the edges of the grid.

    Points move towards their neighbours on the grid, as much as they are close
    to them in the grid and in space.

    Attributes:
        n_grid: points on each side of the grid.
        threshold: smallest weight of an edge, below which it is ignored.
        eps_sqr: squared distance scale of the weights.
        lam: strength of the smoothing.
    """

    n_grid: Final[int]
    threshold: Final[float]
    eps_sqr: Final[float]
    lam: Final[float]

    def __init__(self, n_grid: int, *, threshold: float = 1e-12, eps_sqr: float = 0.02**2, lam: float = 0.5) -> None:
        """Initialize.

        Args:
            n_grid: points on each side of the grid.
            threshold: smallest weight of an edge, below which it is ignored.
            eps_sqr: squared distance scale of the weights.
            lam: strength of the smoothing.
        """
        super().__init__()
        self.n_grid = n_grid
        self.threshold = threshold
        self.eps_sqr = eps_sqr
        self.lam = lam
        return

    def forward(self, pc: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        """Filter the coordinates ``(B, 3, M)`` folded from the grid ``(B, 2, M)``."""
        batch_size = pc.shape[0]
        grid_exp = grid.view(batch_size, 2, self.n_grid, self.n_grid)
        pc_exp = pc.view(batch_size, 3, self.n_grid, self.n_grid)
        graph_feature = torch.cat((grid_exp, pc_exp), 1).contiguous()
        delta_x = graph_feature[:, :, :-1, :] - graph_feature[:, :, 1:, :]
        delta_y = graph_feature[:, :, :, :-1] - graph_feature[:, :, :, 1:]
        delta_x = torch.exp(-torch.sum(delta_x**2, dim=1) / self.eps_sqr)
        delta_y = torch.exp(-torch.sum(delta_y**2, dim=1) / self.eps_sqr)
        delta_x = (delta_x > self.threshold).float() * delta_x
        delta_y = (delta_y > self.threshold).float() * delta_y
        delta_x_left = F.pad(delta_x, pad=[0, 0, 1, 0])
        delta_x_right = F.pad(delta_x, pad=[0, 0, 0, 1])
        delta_y_top = F.pad(delta_y, pad=[1, 0])
        delta_y_bottom = F.pad(delta_y, pad=[0, 1])
        delta = torch.stack((delta_x_left, delta_x_right, delta_y_top, delta_y_bottom), dim=1)
        delta_sum = torch.sum(delta, dim=1).unsqueeze(1).expand(-1, 3, -1, -1)
        delta_x_exp = delta_x.unsqueeze(1).expand(-1, 3, -1, -1)
        delta_y_exp = delta_y.unsqueeze(1).expand(-1, 3, -1, -1)
        pc_filter1 = F.pad(pc_exp[:, :, :-1, :] * delta_x_exp, [0, 0, 1, 0])
        pc_filter1 += F.pad(pc_exp[:, :, 1:, :] * delta_x_exp, [0, 0, 0, 1])
        pc_filter1 += F.pad(pc_exp[:, :, :, :-1] * delta_y_exp, [1, 0])
        pc_filter1 += F.pad(pc_exp[:, :, :, 1:] * delta_y_exp, [0, 1])
        pc_filter = (1 - self.lam * delta_sum) * pc_exp + self.lam * pc_filter1
        return pc_filter.view(batch_size, 3, -1)

    @override
    def extra_repr(self) -> str:
        return f'n_grid={self.n_grid}, threshold={self.threshold}, eps_sqr={self.eps_sqr}, lam={self.lam}'


class FullyConnected(BasePointDecoder):
    """Fully connected baseline decoder for fixed point counts."""

    m: Final[int]
    mlp: nn.Sequential

    def __init__(self) -> None:
        """Initialize."""
        super().__init__()
        self.h_dim = [256, 256]
        self.m = self.m_training
        self.mlp = nn.Sequential(
            LinearLayer(self.w_dim, self.h_dim[0], batch_norm=False, act_cls=nn.ReLU),
            LinearLayer(self.h_dim[0], self.h_dim[1], batch_norm=False, act_cls=nn.ReLU),
            LinearLayer(self.h_dim[1], OUT_CHAN * self.m, batch_norm=False, act_cls=None),
            nn.Unflatten(1, (OUT_CHAN, self.m)),
        )
        return

    def forward(
        self,
        w: torch.Tensor,
        m: int,
        s: torch.Tensor,
    ) -> Decoded:
        """Decode codeword."""
        return Decoded(cloud=self.mlp(w))


class FoldingBlock(nn.Module):
    """Folding layer transforming grid coordinates concatenated with codewords."""

    join: AppendCodeword
    layers: nn.Sequential

    def __init__(self, in_channel: int, h_dim: list[int], out_dim: int) -> None:
        """Initialize."""
        super().__init__()
        self.join = AppendCodeword()
        modules: list[nn.Module] = [nn.Conv1d(in_channel, h_dim[0], kernel_size=1)]
        for in_d, out_d in itertools.pairwise((*h_dim, out_dim)):
            modules.extend([nn.ReLU(inplace=True), nn.Conv1d(in_d, out_d, kernel_size=1)])

        self.layers = nn.Sequential(*modules)
        return

    def forward(self, grids: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Fold the grid ``(B, D, M)`` conditioned on the codeword ``(B, W)``."""
        return self.layers(self.join(grids, w))


class FoldingNet(BasePointDecoder):
    """FoldingNet 2D grid deformation decoder."""

    m: Final[int]
    num_grid: Final[int]
    m_grid: Final[int]
    grid: nn.Parameter
    fold1: FoldingBlock
    fold2: FoldingBlock
    filter: GridFilter | None

    def __init__(self) -> None:
        """Initialize."""
        super().__init__()
        self.h_dim = [512] * 4
        self.m = self.m_training
        self.num_grid = round(float(np.sqrt(self.m)))
        self.m_grid = self.num_grid**2
        xx = torch.linspace(-0.3, 0.3, self.num_grid, dtype=torch.float)
        yy = torch.linspace(-0.3, 0.3, self.num_grid, dtype=torch.float)
        grid_mesh = torch.stack(torch.meshgrid(xx, yy, indexing='ij')).view(2, -1)
        self.grid = nn.Parameter(grid_mesh, requires_grad=False)
        self.fold1 = FoldingBlock(self.w_dim + 2, self.h_dim[0:2], OUT_CHAN)
        self.fold2 = FoldingBlock(self.w_dim + 3, self.h_dim[2:4], OUT_CHAN)
        self.filter = GridFilter(self.num_grid) if self.filtering else None
        return

    def forward(
        self,
        w: torch.Tensor,
        m: int,
        s: torch.Tensor,
    ) -> Decoded:
        """Decode codeword by folding grid."""
        grid = self.grid.unsqueeze(0).repeat(w.size(0), 1, 1)
        return Decoded(cloud=self._fold(grid, w))

    def _fold(self, grid: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        x = self.fold2(self.fold1(grid, w), w)
        if self.filter is not None:
            x = self.filter(x, grid)

        return x


class TearingNet(FoldingNet):
    """TearingNet decoder learning topological cuts on 2D grid."""

    to_grid: nn.Unflatten
    join: AppendCodeword
    tearing1: nn.Sequential
    tearing2: nn.Sequential

    def __init__(self) -> None:
        """Initialize."""
        super().__init__()
        self.h_dim.extend([512, 512, 64, 512, 512, 2])
        self.to_grid = nn.Unflatten(2, (self.num_grid, self.num_grid))
        self.join = AppendCodeword()
        modules1: list[nn.Module] = [nn.Conv2d(self.w_dim + 5, self.h_dim[4], kernel_size=1)]
        for in_d, out_d in itertools.pairwise(self.h_dim[4:7]):
            modules1.extend([nn.ReLU(inplace=True), nn.Conv2d(in_d, out_d, kernel_size=1)])

        self.tearing1 = nn.Sequential(*modules1)
        modules2: list[nn.Module] = [nn.Conv2d(self.w_dim + 5 + self.h_dim[6], self.h_dim[7], kernel_size=1)]
        for in_d, out_d in itertools.pairwise(self.h_dim[7:10]):
            modules2.extend([nn.ReLU(inplace=True), nn.Conv2d(in_d, out_d, kernel_size=1)])

        modules2.append(nn.Flatten(2))
        self.tearing2 = nn.Sequential(*modules2)
        return

    def forward(
        self,
        w: torch.Tensor,
        m: int,
        s: torch.Tensor,
    ) -> Decoded:
        """Decode with two-stage grid tearing."""
        grid = self.grid.unsqueeze(0).repeat(w.size(0), 1, 1)
        x = self._fold(grid, w)
        features = self.join(torch.cat((self.to_grid(grid), self.to_grid(x)), dim=1), w)
        grid = grid + self.tearing2(torch.cat((features, self.tearing1(features)), dim=1))
        x = self._fold(grid, w)
        if self.filter is not None:
            x = self.filter(x, grid)

        return Decoded(cloud=x)


class AtlasNetv2(BasePointDecoder):
    """AtlasNet base patch decoder."""

    patch_embed_dim: int = 2
    num_patches: Final[int]
    embedding_dim: Final[int]
    patches: nn.ModuleList
    deformations: nn.ModuleList
    join: AppendCodeword
    filter: nn.Module

    def __init__(self) -> None:
        """Initialize."""
        super().__init__()
        self.num_patches = self.n_components
        self.embedding_dim = self.w_dim + self.patch_embed_dim
        self.h_dim = [128]
        self.patches = nn.ModuleList([self.get_mlp_adj() for _ in range(self.num_patches)])
        self.deformations = nn.ModuleList([nn.Identity() for _ in range(self.num_patches)])
        self.join = AppendCodeword()
        self.filter = GraphFilter() if self.filtering else nn.Identity()
        return

    def forward(
        self,
        w: torch.Tensor,
        m: int,
        s: torch.Tensor,
    ) -> Decoded:
        """Decode codeword into patched point cloud."""
        patch_pts = m // self.num_patches
        outs: list[torch.Tensor] = []
        for deformation, patch in zip(self.deformations, self.patches, strict=True):
            rand_grid = torch.rand(w.size(0), 2, patch_pts, device=w.device)
            outs.append(patch(self.join(deformation(rand_grid), w)))

        return Decoded(cloud=self.filter(torch.cat(outs, 2).contiguous()))

    def get_mlp_adj(self) -> nn.Sequential:
        """Build multi-layer perceptron for patch mapping."""
        dims = (self.embedding_dim, self.embedding_dim, self.embedding_dim // 2, self.embedding_dim // 4)
        modules = [
            PointsConvLayer(in_dim, out_dim, act_cls=nn.ReLU, bn_momentum=self.bn_momentum)
            for in_dim, out_dim in itertools.pairwise(dims)
        ]
        modules.append(PointsConvLayer(dims[-1], OUT_CHAN, batch_norm=False, act_cls=nn.Tanh))
        return nn.Sequential(*modules)


class AtlasNetv2Deformation(AtlasNetv2):
    """AtlasNet patch deformation variant."""

    patch_embed_dim = 10

    def __init__(self) -> None:
        """Initialize."""
        super().__init__()
        self.deformations = nn.ModuleList([self.get_patch_deformation() for _ in range(self.num_patches)])
        return

    def get_patch_deformation(self) -> nn.Module:
        """Build patch deformation module."""
        dim = self.h_dim[0]
        return nn.Sequential(
            PointsConvLayer(2, dim, act_cls=nn.ReLU, bn_momentum=self.bn_momentum),
            PointsConvLayer(dim, dim, act_cls=nn.ReLU, bn_momentum=self.bn_momentum),
            PointsConvLayer(dim, self.patch_embed_dim, batch_norm=False, act_cls=nn.Tanh),
        )


class AtlasNetV2Translation(AtlasNetv2):
    """AtlasNet patch translation variant with fixed grid parameter."""

    patch_embed_dim = 10
    m_patch: Final[int]
    grid: nn.Parameter

    def __init__(self) -> None:
        """Initialize."""
        super().__init__()
        self.m_patch = self.m_training // self.num_patches
        grid_data = torch.rand(self.num_patches, 2, self.m_patch)
        self.grid = nn.Parameter(grid_data)
        self.grid.data = F.pad(self.grid.data, (0, 0, 0, self.patch_embed_dim - 2))
        return

    def forward(
        self,
        w: torch.Tensor,
        m: int,
        s: torch.Tensor,
    ) -> Decoded:
        """Decode with translated grid patches."""
        outs: list[torch.Tensor] = []
        for grid, patch in zip(self.grid, self.patches, strict=True):
            outs.append(patch(self.join(grid.expand(w.size(0), -1, -1), w)))

        return Decoded(cloud=self.filter(torch.cat(outs, 2).contiguous()))


class ComponentAttention(nn.Module):
    """Choose, for every point, how much each component contributes.

    While training, Gumbel noise makes the choice explore the components;
    evaluation uses the softmax without noise.

    Attributes:
        score: layer scoring every component from the features of all of them.
        temperature: temperature of the softmax.
    """

    score: PointsConvLayer
    temperature: Final[float]

    def __init__(self, in_dim: int, n_components: int, *, temperature: float) -> None:
        """Initialize.

        Args:
            in_dim: features of all the components together.
            n_components: components to choose among.
            temperature: temperature of the softmax.
        """
        super().__init__()
        self.score = PointsConvLayer(in_dim, n_components, batch_norm=False, act_cls=None)
        self.temperature = temperature
        return

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return the share of every component for every point ``(B, M, C)``."""
        scores = self.score(features)
        if self.training:
            return F.gumbel_softmax(scores, tau=self.temperature, dim=1).transpose(2, 1)

        return F.softmax(scores / self.temperature, dim=1).transpose(2, 1)

    @override
    def extra_repr(self) -> str:
        return f'temperature={self.temperature}'


class PCGenBase(BasePointDecoder):
    """Base decoder for Point Cloud Generator."""

    concat: bool = False
    sample_dim: Final[int]
    attention_temperature: Final[float]
    sphere: SphereProjection
    map_sample: nn.Sequential
    join: nn.Module
    group_conv: nn.ModuleList
    group_final: nn.ModuleList
    attention: ComponentAttention | None
    filter: nn.Module

    def __init__(self) -> None:
        """Initialize."""
        super().__init__()
        cfg = Experiment.get_config().autoencoding.architecture.decoder
        self.h_dim = list(cfg.hidden_dims)
        self.sample_dim = cfg.sample_dim
        self.attention_temperature = cfg.attention_temperature
        self.sphere = SphereProjection()
        self.map_sample = self._get_map_sample()
        self.join = self.get_join()
        self.group_conv = nn.ModuleList()
        self.group_final = nn.ModuleList()
        if self.concat:
            self.h_dim[1] = 2 * self.h_dim[1]

        for _ in range(self.n_components):
            self.group_conv.append(self._get_component_conv())
            self.group_final.append(PointsConvLayer(self.h_dim[-1], OUT_CHAN, batch_norm=False, act_cls=None))

        self.attention = self._get_attention()
        self.filter = GraphFilter() if self.filtering else nn.Identity()
        return

    def draw_sampling(self, n_clouds: int, m: int, /, *, loop: bool) -> torch.Tensor:
        """Draw the sampling of the generating surface, optionally followed by a closed path through its corners.

        The path, a loop through the corners of the sampling space, shows how the decoder folds the surface.

        Args:
            n_clouds: the number of clouds.
            m: the number of random points of each cloud.
            loop: whether to append the path after the random points.

        Returns:
            The sampling, of shape ``(n_clouds, sample_dim, m)``, longer with the path.
        """
        device = next(self.parameters()).device
        sampling = torch.randn(n_clouds, self.sample_dim, m, device=device)
        if not loop:
            return sampling

        corners = torch.cat(
            [torch.eye(self.sample_dim, device=device), -torch.eye(self.sample_dim, device=device)], dim=1
        )
        box = corners.unsqueeze(0).expand(n_clouds, -1, -1)
        steps = [step * box.roll(1, dims=2) + (1 - step) * box for step in torch.arange(0, 1, 0.03)]
        return torch.cat([sampling, *steps], dim=2)

    def forward(
        self,
        w: torch.Tensor,
        m: int,
        s: torch.Tensor,
    ) -> Decoded:
        """Decode spherical sampling points and codeword into 3D points."""
        x = s if s.numel() else self.draw_sampling(w.size(0), m, loop=False)
        x = self.join(self.map_sample(self.sphere(x)), w)
        features = [conv(x) for conv in self.group_conv]
        components = torch.stack([final(f) for final, f in zip(self.group_final, features, strict=True)], dim=3)
        if self.attention is None:
            return Decoded(cloud=self.filter(components.squeeze(3)))

        attention = self.attention(torch.cat(features, dim=1).contiguous())
        cloud = (components * attention.unsqueeze(1)).sum(3)
        return Decoded(cloud=self.filter(cloud), attention=attention, components=components)

    @abc.abstractmethod
    def get_map2_act(self) -> ActClass:
        """Resolve activation for the second spherical mapping layer."""

    def _get_attention(self) -> ComponentAttention | None:
        if self.n_components == 1:
            return None

        total_dim = self.h_dim[-1] * self.n_components
        return ComponentAttention(total_dim, self.n_components, temperature=self.attention_temperature)

    def _get_component_conv(self) -> nn.Sequential:
        modules = [
            PointsConvLayer(in_dim, out_dim, act_cls=nn.ReLU, bn_momentum=self.bn_momentum)
            for in_dim, out_dim in itertools.pairwise(self.h_dim[1:])
        ]
        return nn.Sequential(*modules)

    def _get_map_sample(self) -> nn.Sequential:
        return nn.Sequential(
            PointsConvLayer(self.sample_dim, self.h_dim[0], batch_norm=False, act_cls=nn.ReLU),
            PointsConvLayer(self.h_dim[0], self.h_dim[1], batch_norm=False, act_cls=self.get_map2_act()),
        )

    @staticmethod
    @abc.abstractmethod
    def get_join() -> nn.Module:
        """Get the module combining the sampling representation and the codeword."""


class PCGen(PCGenBase):
    """PCGen decoder with multiplicative codeword conditioning."""

    @override
    def get_map2_act(self) -> ActClass:
        """Resolve activation for the second spherical mapping layer."""
        return nn.Hardtanh

    @staticmethod
    @override
    def get_join() -> nn.Module:
        """Multiply sampling representation by codeword."""
        return Modulate()


class PCGenConcat(PCGenBase):
    """PCGen decoder with concatenated codeword conditioning."""

    concat = True

    @override
    def get_map2_act(self) -> ActClass:
        """Resolve activation for the second spherical mapping layer."""
        return self.act_cls

    @staticmethod
    @override
    def get_join() -> nn.Module:
        """Concatenate expanded codeword with sampling representation."""
        return PrependCodeword()


def get_decoder() -> BasePointDecoder:
    """Instantiate the decoder specified in the configuration."""
    cfg = Experiment.get_config().autoencoding.architecture
    decoder_dict: dict[Decoders, type[BasePointDecoder]] = {
        Decoders.ATLASNET_DEFORMATION: AtlasNetv2Deformation,
        Decoders.ATLASNET_TRANSLATION: AtlasNetV2Translation,
        Decoders.FOLDINGNET: FoldingNet,
        Decoders.FULL: FullyConnected,
        Decoders.PCGEN: PCGen,
        Decoders.PCGEN_CONCAT: PCGenConcat,
        Decoders.TEARINGNET: TearingNet,
    }
    return decoder_dict[cfg.decoder.name]()
