"""Point cloud encoders."""

import abc
import itertools

from typing import Final, override

import torch
import torch.nn as nn

from src.config.experiment import Experiment
from src.config.options import Encoders
from src.config.torch import ActClass
from src.data.structures import IN_CHAN
from src.module.layers import (
    EdgeConvLayer,
    EdgeFeatures,
    GraphMaxPooling,
    LinearLayer,
    LocalCovariance,
    Max,
    Mean,
    NearestNeighbours,
    PointsConvLayer,
    Transpose,
)

__all__ = [
    'DGCNN',
    'LDGCNN',
    'BasePointEncoder',
    'BaseWEncoder',
    'FoldingNet',
    'WEncoder',
    'get_encoder',
]


class BaseWEncoder(nn.Module, abc.ABC):
    """Abstract base class for codeword encoders."""

    w_dim: Final[int]
    embedding_dim: Final[int]
    dim_codes: Final[int]
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
        self.act_cls = cfg.autoencoding.architecture.act_cls
        self.bn_momentum = cfg.autoencoding.architecture.bn_momentum
        cfg_w = cfg.w_autoencoding
        assert cfg_w is not None
        self.z_dim = cfg_w.model.z_dim
        self.dropout = cfg_w.architecture.dropout
        self.h_dims = list(cfg_w.architecture.encoder_h_dims)
        self.h_dim = self.h_dims
        return

    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input into mean and variance representations."""
        ...


class WEncoder(BaseWEncoder):
    """Codeword encoder mapping quantized representations to continuous latents."""

    layers: nn.Sequential

    def __init__(self) -> None:
        """Initialize."""
        super().__init__()
        self.layers = nn.Sequential(
            *self._get_conv_layers(),
            nn.Flatten(),
            nn.Dropout(self.dropout),
            LinearLayer(self.w_dim * self.h_dims[-1], self.w_dim, act_cls=self.act_cls, bn_momentum=self.bn_momentum),
            nn.Dropout(self.dropout),
            LinearLayer(self.w_dim, 2 * self.z_dim, act_cls=None),
        )
        return

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input into mean and variance representations."""
        return self.layers(x)

    def _get_conv_layers(self) -> list[nn.Module]:
        return [
            PointsConvLayer(
                self.embedding_dim * in_d,
                self.embedding_dim * out_d,
                act_cls=self.act_cls,
                bn_momentum=self.bn_momentum,
            )
            for in_d, out_d in itertools.pairwise((1, *self.h_dims))
        ]


class BasePointEncoder(nn.Module, abc.ABC):
    """Abstract base class for point cloud encoders."""

    n_neighbours: Final[int]
    w_dim: Final[int]
    act_cls: Final[ActClass]
    bn_momentum: Final[float]
    h_dim: list[int]

    def __init__(self) -> None:
        """Initialize."""
        super().__init__()
        cfg_ae = Experiment.get_config().autoencoding
        self.n_neighbours = cfg_ae.architecture.encoder.n_neighbours
        self.w_dim = cfg_ae.model.w_dim
        self.act_cls = cfg_ae.architecture.act_cls
        self.bn_momentum = cfg_ae.architecture.bn_momentum
        return

    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode point cloud coordinates ``(B, N, 3)`` into codeword ``(B, W)``."""
        ...


class LDGCNN(BasePointEncoder):
    """Large Dynamic Graph CNN encoder."""

    transpose: Transpose
    neighbours: NearestNeighbours
    edges: EdgeFeatures
    edge_block: nn.Sequential
    pool: GraphMaxPooling
    points_convs: nn.Sequential
    head: nn.Sequential

    def __init__(self) -> None:
        """Initialize."""
        super().__init__()
        self.h_dim = [64, 64, 128, 256]
        self.transpose = Transpose(2, 1)
        self.neighbours = NearestNeighbours(self.n_neighbours)
        self.edges = EdgeFeatures(self.n_neighbours)
        self.edge_block = self._get_edge_block()
        self.pool = GraphMaxPooling(self.n_neighbours)
        self.points_convs = self._get_points_convs()
        self.head = self._get_head()
        return

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode point cloud coordinates ``(B, N, 3)``."""
        x = self.transpose(x)
        indices = self.neighbours(x)
        x = self.edge_block(self.edges(x, indices))
        xs = [x]
        # the graph of the coordinates is kept, which is what links the layers
        for conv in self.points_convs:
            x = conv(self.pool(x, indices))
            xs.append(x)

        return self.head(torch.cat(xs, dim=1).contiguous())

    def _get_edge_block(self) -> nn.Sequential:
        return nn.Sequential(
            EdgeConvLayer(2 * IN_CHAN, self.h_dim[0], act_cls=self.act_cls, bn_momentum=self.bn_momentum),
            Max(dim=3),
        )

    def _get_head(self) -> nn.Sequential:
        return nn.Sequential(
            PointsConvLayer(sum(self.h_dim), self.w_dim, act_cls=None, batch_norm=False),
            Max(dim=2),
        )

    def _get_points_convs(self) -> nn.Sequential:
        modules = [
            PointsConvLayer(in_dim, out_dim, act_cls=self.act_cls, bn_momentum=self.bn_momentum)
            for in_dim, out_dim in itertools.pairwise(self.h_dim)
        ]
        return nn.Sequential(*modules)


class DGCNN(BasePointEncoder):
    """Dynamic Graph CNN encoder with dynamic neighbourhood recomputation."""

    transpose: Transpose
    blocks: nn.ModuleList
    head: nn.Sequential

    def __init__(self) -> None:
        """Initialize."""
        super().__init__()
        self.h_dim = [64, 64, 128, 256]
        self.transpose = Transpose(2, 1)
        self.blocks = self._get_blocks()
        self.head = nn.Sequential(nn.Conv1d(sum(self.h_dim), self.w_dim, kernel_size=1), Max(dim=2))
        return

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode point cloud coordinates ``(B, N, 3)``."""
        x = self.transpose(x)
        xs: list[torch.Tensor] = []
        # the graph is recomputed from the features of every block
        for block in self.blocks:
            x = block(x)
            xs.append(x)

        return self.head(torch.cat(xs, dim=1).contiguous())

    def _get_blocks(self) -> nn.ModuleList:
        return nn.ModuleList(
            nn.Sequential(
                EdgeFeatures(self.n_neighbours),
                EdgeConvLayer(2 * in_dim, out_dim, batch_norm=True, act_cls=self.act_cls, bn_momentum=self.bn_momentum),
                Max(dim=3),
            )
            for in_dim, out_dim in itertools.pairwise((IN_CHAN, *self.h_dim))
        )


class FoldingNet(BasePointEncoder):
    """FoldingNet encoder using local covariance descriptors."""

    layers: nn.Sequential

    def __init__(self) -> None:
        """Initialize."""
        super().__init__()
        self.h_dim = [64, 64, 64, 128, 1024, 1024]
        self.layers = nn.Sequential(
            Transpose(2, 1),
            LocalCovariance(self.n_neighbours),
            self._get_point_mlp(),
            GraphMaxPooling(16),
            PointsConvLayer(self.h_dim[2], self.h_dim[3], act_cls=nn.ReLU, batch_norm=False),
            GraphMaxPooling(16),
            PointsConvLayer(self.h_dim[3], self.h_dim[4], act_cls=nn.ReLU, batch_norm=False),
            Mean(dim=-1),
            nn.Sequential(
                LinearLayer(self.h_dim[4], self.h_dim[5], act_cls=nn.ReLU, batch_norm=False),
                nn.Linear(self.h_dim[5], self.w_dim),
            ),
        )
        return

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode point cloud coordinates ``(B, N, 3)``."""
        return self.layers(x)

    def _get_point_mlp(self) -> nn.Sequential:
        modules = [
            PointsConvLayer(in_dim, out_dim, act_cls=nn.ReLU, batch_norm=False)
            for in_dim, out_dim in itertools.pairwise((IN_CHAN + IN_CHAN**2, *self.h_dim[:3]))
        ]
        return nn.Sequential(*modules)


def get_encoder() -> BasePointEncoder:
    """Instantiate the encoder specified in the configuration."""
    cfg = Experiment.get_config().autoencoding.architecture
    dict_encoder: dict[Encoders, type[BasePointEncoder]] = {
        Encoders.DGCNN: DGCNN,
        Encoders.FOLDINGNET: FoldingNet,
        Encoders.LDGCNN: LDGCNN,
    }
    return dict_encoder[cfg.encoder.name]()
