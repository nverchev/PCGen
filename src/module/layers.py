"""Neural network layer building blocks and point cloud operators."""

import abc
import itertools
import math
import warnings

from collections.abc import Sequence
from typing import Any, Final, Generic, Protocol, TypeVar, override, runtime_checkable

import torch
import torch.nn as nn
from torch import Tensor
from torch.autograd import Function

from src.config.environment import DEBUG_MODE
from src.config.torch import ActClass, NormClass
from src.utils.neighbour_ops import get_graph_features, get_local_covariance, graph_filtering, graph_max_pooling, knn

DEFAULT_BN_MOMENTUM: Final[float] = 0.1
BN_MOMENTUM: Final[float] = DEFAULT_BN_MOMENTUM

__all__ = [
    'BN_MOMENTUM',
    'DEFAULT_BN_MOMENTUM',
    'ActClass',
    'AppendCodeword',
    'BaseLayer',
    'BaseResBlock',
    'EdgeConvLayer',
    'EdgeConvResBlock',
    'EdgeFeatures',
    'GraphFilter',
    'GraphMaxPooling',
    'HasInplace',
    'HasResetParamPrivate',
    'HasResetParamPublic',
    'LinearLayer',
    'LinearResBlock',
    'LocalCovariance',
    'Max',
    'Mean',
    'Modulate',
    'NearestNeighbours',
    'NormClass',
    'PointsConvLayer',
    'PointsConvResBlock',
    'PrependCodeword',
    'ProjectionLayer',
    'Repeat',
    'SphereProjection',
    'TransferGrad',
    'Transpose',
    'View',
    'Weighted',
    'debug_check',
    'reset_parameters_recursive',
]

type _grad_t = tuple[torch.Tensor, ...] | torch.Tensor


@runtime_checkable
class HasInplace(Protocol):
    """Protocol for classes that have an inplace option."""

    inplace: bool


@runtime_checkable
class HasResetParamPublic(Protocol):
    """Protocol for classes that support resetting the parameters."""

    def reset_parameters(self) -> None:
        """Reset the parameters of the module."""
        ...


@runtime_checkable
class HasResetParamPrivate(Protocol):
    """Protocol for classes that support resetting the parameters."""

    def _reset_parameters(self) -> None:
        """Reset the parameters of the module."""
        ...


@runtime_checkable
class Weighted(Protocol):
    """Protocol for classes that have a weight attribute."""

    weight: torch.Tensor
    bias: torch.Tensor | None


class View(nn.Module):
    """A simple module that reshapes the input tensor according to the specified shape."""

    shape: Final[tuple[int, ...]]

    def __init__(self, *shape: int) -> None:
        """Initialize."""
        super().__init__()
        self.shape = shape
        return

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return x.view(self.shape)


class Max(nn.Module):
    """Take the largest value along a dimension, which the dimension loses.

    Attributes:
        dim: dimension to pool.
    """

    dim: Final[int]

    def __init__(self, dim: int) -> None:
        """Initialize.

        Args:
            dim: dimension to pool.
        """
        super().__init__()
        self.dim = dim
        return

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pool the largest value."""
        return x.max(dim=self.dim)[0]

    @override
    def extra_repr(self) -> str:
        return f'dim={self.dim}'


class Mean(nn.Module):
    """Average along a dimension, which the dimension loses.

    Attributes:
        dim: dimension to average.
    """

    dim: Final[int]

    def __init__(self, dim: int) -> None:
        """Initialize.

        Args:
            dim: dimension to average.
        """
        super().__init__()
        self.dim = dim
        return

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Average the values."""
        return x.mean(self.dim)

    @override
    def extra_repr(self) -> str:
        return f'dim={self.dim}'


class Repeat(nn.Module):
    """Tile a tensor, as :meth:`torch.Tensor.repeat` does.

    Attributes:
        sizes: times to repeat each dimension.
    """

    sizes: Final[tuple[int, ...]]

    def __init__(self, *sizes: int) -> None:
        """Initialize.

        Args:
            sizes: times to repeat each dimension.
        """
        super().__init__()
        self.sizes = sizes
        return

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Tile the input."""
        return x.repeat(*self.sizes)

    @override
    def extra_repr(self) -> str:
        return f'sizes={self.sizes}'


class Transpose(nn.Module):
    """Swap two dimensions.

    Attributes:
        dim0: first dimension to swap.
        dim1: second dimension to swap.
    """

    dim0: Final[int]
    dim1: Final[int]

    def __init__(self, dim0: int, dim1: int) -> None:
        """Initialize.

        Args:
            dim0: first dimension to swap.
            dim1: second dimension to swap.
        """
        super().__init__()
        self.dim0 = dim0
        self.dim1 = dim1
        return

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Swap the dimensions."""
        return x.transpose(self.dim0, self.dim1)

    @override
    def extra_repr(self) -> str:
        return f'dim0={self.dim0}, dim1={self.dim1}'


class BaseLayer(nn.Module, metaclass=abc.ABCMeta):
    """Base layer with optional batch normalization and activation.

    PCGen preserves PyTorch's default parameter initialisation for linear and convolutional layers,
    applying Kaiming uniform only when the activation is LeakyReLU.
    """

    in_dim: Final[int]
    out_dim: Final[int]
    n_groups_layer: Final[int]
    batch_norm: Final[bool]
    bn_momentum: Final[float]
    use_bias: Final[bool]
    use_trunc_init: Final[bool]
    module: nn.Module
    norm: nn.Module | None
    act: nn.Module | None

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        act_cls: ActClass | None,
        batch_norm: bool = True,
        *,
        bn_momentum: float = DEFAULT_BN_MOMENTUM,
        groups: int | None = None,
        n_groups_layer: int = 1,
        norm_cls: NormClass | None = None,
        use_trunc_init: bool = False,
    ) -> None:
        """Initialize."""
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_groups_layer = groups if groups is not None else n_groups_layer
        self.batch_norm = batch_norm
        self.use_bias = not batch_norm
        self.bn_momentum = bn_momentum
        self.module = self.get_module()
        self.norm = self.get_norm_layer(norm_cls) if batch_norm else None
        self.act = self.get_activation(act_cls)
        self.use_trunc_init = use_trunc_init
        self.reset_parameters()
        if DEBUG_MODE:
            warnings.filterwarnings('ignore', message='.*Full backward hook is firing.*')
            self.register_forward_hook(debug_check)
            self.register_full_backward_hook(debug_check)

        return

    def reset_parameters(self) -> None:
        """(Re)-Initialize weights and bias.

        Leaves PyTorch default initialization alone unless activation is LeakyReLU,
        in which case Kaiming uniform with its negative slope is applied.
        """
        if self.use_trunc_init:
            if isinstance(self.module, Weighted):
                nn.init.trunc_normal_(self.module.weight, mean=0.0, std=0.02, a=-0.04, b=0.04)

            if self.norm is None and isinstance(self.module, Weighted) and self.module.bias is not None:
                nn.init.zeros_(self.module.bias)
        else:
            _reset_parameters_or_warn(self.module)

            if self.act is not None and isinstance(self.act, nn.LeakyReLU):
                if isinstance(self.module, Weighted):
                    nn.init.kaiming_uniform_(self.module.weight, a=self.act.negative_slope)

        if self.norm is not None:
            _reset_parameters_or_warn(self.norm)

        return

    @abc.abstractmethod
    def get_module(self) -> nn.Module:
        """Return the wrapped dense or convolution layer."""

    @abc.abstractmethod
    def get_norm_layer(self, norm_cls: NormClass | None) -> nn.Module | None:
        """Return normalization layer."""

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply dense/convolution projection, normalization, and activation."""
        y = self.module(x)
        if self.norm is not None:
            y = self.norm(y)

        if self.act is not None:
            y = self.act(y)

        return y

    @staticmethod
    def get_activation(act_cls: ActClass | None) -> nn.Module | None:
        """Get activation function, setting inplace=True except for Tanh."""
        if act_cls is None:
            return None

        act = act_cls()
        if isinstance(act, HasInplace) and not isinstance(act, nn.Tanh):
            act.inplace = True

        return act


class LinearLayer(BaseLayer):
    """Fully connected layer with optional batch normalization and activation."""

    @override
    def get_module(self) -> nn.Module:
        """Return linear projection."""
        if self.n_groups_layer > 1:
            msg = 'nn.Linear has no option for groups'
            raise NotImplementedError(msg)

        return nn.Linear(self.in_dim, self.out_dim, bias=self.use_bias)

    @override
    def get_norm_layer(self, norm_cls: NormClass | None) -> nn.Module | None:
        """Return 1D batch normalization."""
        if norm_cls is not None:
            return norm_cls(self.out_dim)

        return nn.BatchNorm1d(self.out_dim, momentum=self.bn_momentum)


class PointsConvLayer(BaseLayer):
    """1D convolution treating point channels as feature channels."""

    @override
    def get_module(self) -> nn.Module:
        """Return Conv1d layer with kernel size 1."""
        return nn.Conv1d(
            self.in_dim,
            self.out_dim,
            kernel_size=1,
            bias=self.use_bias,
            groups=self.n_groups_layer,
        )

    @override
    def get_norm_layer(self, norm_cls: NormClass | None) -> nn.Module | None:
        """Return 1D batch normalization."""
        if norm_cls is not None:
            return norm_cls(self.out_dim)

        return nn.BatchNorm1d(self.out_dim, momentum=self.bn_momentum)


class EdgeConvLayer(BaseLayer):
    """2D convolution over point pairs/edges."""

    @override
    def get_module(self) -> nn.Module:
        """Return Conv2d layer with kernel size 1."""
        return nn.Conv2d(
            self.in_dim,
            self.out_dim,
            kernel_size=1,
            bias=self.use_bias,
            groups=self.n_groups_layer,
        )

    @override
    def get_norm_layer(self, norm_cls: NormClass | None) -> nn.Module | None:
        """Return 2D batch normalization."""
        if norm_cls is not None:
            return norm_cls(self.out_dim)

        return nn.BatchNorm2d(self.out_dim, momentum=self.bn_momentum)


class AppendCodeword(nn.Module):
    """Condition point features on the codeword by appending it to every point."""

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Concatenate the codeword ``(B, W)`` after the features ``(B, H, ...)``."""
        w_exp = w.view(*w.shape, *[1] * (x.dim() - 2)).expand(-1, -1, *x.shape[2:])
        return torch.cat((x, w_exp), dim=1)


class EdgeFeatures(nn.Module):
    """Build the edge features between every point and its neighbours.

    Attributes:
        n_neighbours: neighbours of every point.
    """

    n_neighbours: int

    def __init__(self, n_neighbours: int = 20) -> None:
        """Initialize.

        Args:
            n_neighbours: neighbours of every point.
        """
        super().__init__()
        self.n_neighbours = n_neighbours
        return

    def forward(self, x: torch.Tensor, indices: torch.Tensor | None = None) -> torch.Tensor:
        """Return the edge features ``(B, 2C, N, n_neighbours)`` of the features ``(B, C, N)``.

        Args:
            x: features ``(B, C, N)``.
            indices: neighbours to build the edges on, computed from ``x`` when
                left out.
        """
        return get_graph_features(x, self.n_neighbours, indices)[1]

    def extra_repr(self) -> str:
        return f'n_neighbours={self.n_neighbours}'


class GraphFilter(nn.Module):
    """Sharpen a cloud by moving every point away from its nearest neighbours.

    Attributes:
        n_neighbours: neighbours of every point, the point itself included.
    """

    n_neighbours: int

    def __init__(self, n_neighbours: int = 4) -> None:
        """Initialize.

        Args:
            n_neighbours: neighbours of every point, the point itself included.
        """
        super().__init__()
        self.n_neighbours = n_neighbours
        return

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Filter the coordinates ``(B, 3, M)``."""
        return graph_filtering(x, self.n_neighbours)

    def extra_repr(self) -> str:
        return f'n_neighbours={self.n_neighbours}'


class GraphMaxPooling(nn.Module):
    """Pool the largest feature value over the neighbours.

    Attributes:
        n_neighbours: neighbours of every point.
    """

    n_neighbours: int

    def __init__(self, n_neighbours: int = 16) -> None:
        """Initialize.

        Args:
            n_neighbours: neighbours of every point.
        """
        super().__init__()
        self.n_neighbours = n_neighbours
        return

    def forward(self, x: torch.Tensor, indices: torch.Tensor | None = None) -> torch.Tensor:
        """Return the pooled features ``(B, C, N)``.

        Args:
            x: features ``(B, C, N)``.
            indices: neighbours to pool over, computed from ``x`` when left out.
        """
        return graph_max_pooling(x, self.n_neighbours, indices)

    def extra_repr(self) -> str:
        return f'n_neighbours={self.n_neighbours}'


class LocalCovariance(nn.Module):
    """Append to every point the covariance of its neighbours.

    Attributes:
        n_neighbours: neighbours of every point.
    """

    n_neighbours: int

    def __init__(self, n_neighbours: int = 16) -> None:
        """Initialize.

        Args:
            n_neighbours: neighbours of every point.
        """
        super().__init__()
        self.n_neighbours = n_neighbours
        return

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the coordinates ``(B, 3, N)`` followed by the covariances ``(B, 9, N)``."""
        return get_local_covariance(x, self.n_neighbours)

    def extra_repr(self) -> str:
        return f'n_neighbours={self.n_neighbours}'


class Modulate(nn.Module):
    """Condition point features on the codeword by multiplying them channel by channel."""

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Multiply the features ``(B, W, M)`` by the codeword ``(B, W)``."""
        return w.unsqueeze(2) * x


class NearestNeighbours(nn.Module):
    """Find the nearest neighbours of every point.

    Attributes:
        n_neighbours: neighbours of every point.
    """

    n_neighbours: int

    def __init__(self, n_neighbours: int = 20) -> None:
        """Initialize.

        Args:
            n_neighbours: neighbours of every point.
        """
        super().__init__()
        self.n_neighbours = n_neighbours
        return

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the indices ``(B, N, n_neighbours)`` of the neighbours in the features ``(B, C, N)``."""
        return knn(x, self.n_neighbours)

    def extra_repr(self) -> str:
        return f'n_neighbours={self.n_neighbours}'


class PrependCodeword(nn.Module):
    """Condition point features on the codeword by placing it before the features of every point."""

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """Concatenate the codeword ``(B, W)`` before the features ``(B, H, M)``."""
        return torch.cat((w.unsqueeze(2).expand(-1, -1, x.shape[2]), x), dim=1)


class SphereProjection(nn.Module):
    """Project points onto the unit sphere, along the coordinates."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Divide every point ``(B, D, M)`` by its norm."""
        return x / torch.linalg.vector_norm(x, dim=1, keepdim=True)


class TransferGrad(Function):
    """Straight-through gradient estimator transferring backward gradients."""

    @staticmethod
    @override
    def forward(ctx: Any, *args: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Forward pass returns the quantized tensor unchanged."""
        from_tensor, _to_tensor = args
        return from_tensor

    @staticmethod
    @override
    def backward(ctx: Any, *grad_outputs: torch.Tensor) -> tuple[None, torch.Tensor]:
        """Backward pass routes gradient to the pre-quantized input tensor."""
        return None, grad_outputs[0]


class ProjectionLayer(nn.Module):
    """A layer that projects the input to a different dimension."""

    in_dim: Final[int]
    out_dim: Final[int]

    def __init__(self, in_dim: int, out_dim: int) -> None:
        """Initialize the projection layer."""
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        return

    @override
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        if x.shape[1] != self.in_dim:
            msg = f'Input dimension {x.shape[1]} does not match expected dimension {self.in_dim}'
            raise ValueError(msg)

        if self.in_dim == self.out_dim:
            return x

        if self.in_dim > self.out_dim:
            out = torch.cat([t.mean(1, keepdim=True) for t in torch.chunk(x, self.out_dim, dim=1)], dim=1)
        else:
            out = x.repeat_interleave(self.out_dim // self.in_dim + 1, 1)[:, : self.out_dim, ...]

        var_adjust = math.sqrt(self.in_dim / self.out_dim)
        return out * var_adjust


Layer = TypeVar('Layer', bound=BaseLayer)


class BaseResBlock(nn.Module, Generic[Layer], abc.ABC):
    """A block of a neural network consisting of a sequence of layers."""

    dims: Final[Sequence[int]]
    act_cls: Final[ActClass | None]
    norm_cls: Final[NormClass | None]
    n_groups_layer: Final[int]
    bn_momentum: Final[float]
    layers: nn.ModuleList
    projections: nn.ModuleList

    def __init__(
        self,
        dims: Sequence[int],
        /,
        *,
        act_cls: ActClass | None = None,
        bn_momentum: float = DEFAULT_BN_MOMENTUM,
        n_groups_layer: int = 1,
        norm_cls: NormClass | None = None,
    ) -> None:
        """Initialize."""
        super().__init__()
        self.dims = dims
        self.act_cls = act_cls
        self.bn_momentum = bn_momentum
        self.norm_cls = norm_cls
        self.n_groups_layer = n_groups_layer
        self.layers = nn.ModuleList()
        self.projections = nn.ModuleList()
        for in_dim, out_dim in itertools.pairwise(dims):
            layer = self.get_layer(
                in_dim,
                out_dim,
                act_cls=act_cls,
                bn_momentum=bn_momentum,
                n_groups_layer=n_groups_layer,
                norm_cls=norm_cls,
            )
            self.layers.append(layer)
            if in_dim == out_dim:
                self.projections.append(nn.Identity())
            else:
                self.projections.append(self.get_projection(in_dim, out_dim))

        self._adjust_variance_init()
        return

    @override
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        for i, layer in enumerate(self.layers):
            x = self.projections[i](x) + layer(x)

        return x

    def reset_parameters(self) -> None:
        """(Re)-Initialize the weights and bias of the layer."""
        for layer in self.layers:
            _reset_parameters_or_warn(layer)

        self._adjust_variance_init()
        return

    @classmethod
    @abc.abstractmethod
    def get_layer(
        cls,
        in_dim: int,
        out_dim: int,
        *,
        act_cls: ActClass | None = None,
        bn_momentum: float = DEFAULT_BN_MOMENTUM,
        n_groups_layer: int = 1,
        norm_cls: NormClass | None = None,
    ) -> Layer:
        """Get the layer instance."""

    @classmethod
    def get_projection(cls, in_dim: int, out_dim: int) -> nn.Module:
        """Get a projection layer."""
        return ProjectionLayer(in_dim, out_dim)

    def _adjust_variance_init(self) -> None:
        with torch.no_grad():
            for layer in self.layers:
                wrapped = getattr(layer, 'module', None)
                if isinstance(wrapped, Weighted):
                    wrapped.weight.mul_(1 / math.sqrt(len(self.dims) - 1))
                    if wrapped.bias is not None:
                        wrapped.bias.mul_(0)

        return


class LinearResBlock(BaseResBlock[LinearLayer]):
    """Residual block composed of linear layers."""

    @classmethod
    @override
    def get_layer(
        cls,
        in_dim: int,
        out_dim: int,
        *,
        act_cls: ActClass | None = None,
        bn_momentum: float = DEFAULT_BN_MOMENTUM,
        n_groups_layer: int = 1,
        norm_cls: NormClass | None = None,
    ) -> LinearLayer:
        return LinearLayer(
            in_dim,
            out_dim,
            act_cls=act_cls,
            bn_momentum=bn_momentum,
            n_groups_layer=n_groups_layer,
            norm_cls=norm_cls,
            batch_norm=norm_cls is not None,
        )


class PointsConvResBlock(BaseResBlock[PointsConvLayer]):
    """Residual block composed of point convolution layers."""

    @classmethod
    @override
    def get_layer(
        cls,
        in_dim: int,
        out_dim: int,
        *,
        act_cls: ActClass | None = None,
        bn_momentum: float = DEFAULT_BN_MOMENTUM,
        n_groups_layer: int = 1,
        norm_cls: NormClass | None = None,
    ) -> PointsConvLayer:
        return PointsConvLayer(
            in_dim,
            out_dim,
            act_cls=act_cls,
            bn_momentum=bn_momentum,
            n_groups_layer=n_groups_layer,
            norm_cls=norm_cls,
            batch_norm=norm_cls is not None,
        )


class EdgeConvResBlock(BaseResBlock[EdgeConvLayer]):
    """Residual block composed of edge convolution layers."""

    @classmethod
    @override
    def get_layer(
        cls,
        in_dim: int,
        out_dim: int,
        *,
        act_cls: ActClass | None = None,
        bn_momentum: float = DEFAULT_BN_MOMENTUM,
        n_groups_layer: int = 1,
        norm_cls: NormClass | None = None,
    ) -> EdgeConvLayer:
        return EdgeConvLayer(
            in_dim,
            out_dim,
            act_cls=act_cls,
            bn_momentum=bn_momentum,
            n_groups_layer=n_groups_layer,
            norm_cls=norm_cls,
            batch_norm=norm_cls is not None,
        )


def debug_check(_not_used1: nn.Module, _not_used2: _grad_t, tensor_out: _grad_t) -> None:
    """Validate layer outputs during debug execution."""
    tensor = tensor_out[0] if isinstance(tensor_out, tuple) else tensor_out
    if not torch.isfinite(tensor).all():
        breakpoint()

    return


def _reset_parameters_or_warn(module: nn.Module) -> None:
    if isinstance(module, HasResetParamPublic):
        module.reset_parameters()
        return

    if isinstance(module, HasResetParamPrivate):
        module._reset_parameters()
        return

    if isinstance(module, nn.Module):
        warnings.warn(f'Module {module.__class__.__name__} does not implement reset_parameters().', stacklevel=3)

    return


def reset_parameters_recursive(module: nn.Module, /, *, warn: bool = False) -> None:
    """Recursively reset parameters for all submodules.

    Args:
        module: The module to reset parameters for, along with all its submodules
        warn: Whether to warn about modules without reset methods
    """
    if isinstance(module, HasResetParamPublic):
        module.reset_parameters()
        return

    if isinstance(module, HasResetParamPrivate):
        module._reset_parameters()
        return

    has_params = any(True for _ in module.parameters(recurse=False))
    has_children = any(True for _ in module.children())
    if has_params and not has_children and warn:
        warnings.warn(
            f'Leaf module {module.__class__.__name__} has parameters but does not implement reset_parameters().',
            stacklevel=2,
        )

    for child in module.children():
        reset_parameters_recursive(child, warn=warn)

    return
