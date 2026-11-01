"""Autoencoder model heads and architectures."""

import abc
from typing import Final, override

import torch
import torch.nn as nn

from pykeops.torch import LazyTensor

from src.config.experiment import Experiment
from src.config.options import ModelHeads
from src.data.structures import Inputs, Outputs, Quantization
from src.module.decoders import BasePointDecoder, PCGenBase, get_decoder
from src.module.encoders import BasePointEncoder, get_encoder
from src.module.layers import TransferGrad, reset_parameters_recursive
from src.utils.neighbour_ops import pykeops_square_distance

__all__ = [
    'AE',
    'VQVAE',
    'BaseModel',
    'Oracle',
    'get_autoencoder',
]


class BaseModel(nn.Module, abc.ABC):
    """Abstract base class for point cloud autoencoders."""

    m_training: Final[int]
    m_test: Final[int]

    def __init__(self) -> None:
        """Initialize."""
        super().__init__()
        cfg = Experiment.get_config().autoencoding.train
        self.m_training = cfg.m_training
        self.m_test = cfg.m_test
        return

    @property
    def m(self) -> int:
        """Return point count for training or testing."""
        return self.m_training if self.training else self.m_test

    @abc.abstractmethod
    def forward(self, inputs: Inputs) -> Outputs:
        """Process inputs and produce model outputs."""
        ...

    def get_sampling(self, n_clouds: int, /, *, loop: bool) -> torch.Tensor:
        """Get the sampling of the generating surface for the evaluation, empty when the model draws none.

        Args:
            n_clouds: the number of clouds.
            loop: whether to append a closed path through the corners of the sampling space.

        Returns:
            The sampling, with ``m_test`` random points per cloud.
        """
        return torch.empty(0)

    def reset_parameters(self) -> None:
        """Reset all parameters across submodules."""
        reset_parameters_recursive(self)
        return


class Oracle(BaseModel):
    """Oracle model returning truncated ground-truth cloud directly."""

    @override
    def forward(self, inputs: Inputs) -> Outputs:
        """Slice input cloud to expected point count."""
        return Outputs(inputs.cloud[:, : self.m, :], torch.empty(0), torch.empty(0))


class AE(BaseModel):
    """Standard point cloud autoencoder."""

    encoder: BasePointEncoder
    decoder: BasePointDecoder
    settings: Final[dict[str, list[int]]]

    def __init__(self) -> None:
        """Initialize."""
        super().__init__()
        self.encoder = get_encoder()
        self.decoder = get_decoder()
        self.settings = {'encode_h_dim': self.encoder.h_dim, 'decode_h_dim': self.decoder.h_dim}
        return

    @override
    def forward(self, inputs: Inputs) -> Outputs:
        """Encode input cloud and decode to reconstructed coordinates."""
        return self.decode(self.encode(inputs.cloud), inputs.initial_sampling)

    @override
    def get_sampling(self, n_clouds: int, /, *, loop: bool) -> torch.Tensor:
        if isinstance(self.decoder, PCGenBase):
            return self.decoder.draw_sampling(n_clouds, self.m_test, loop=loop)

        return torch.empty(0)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode point cloud to codeword."""
        return self.encoder(x)

    def decode(self, w: torch.Tensor, /, initial_sampling: torch.Tensor | None = None) -> Outputs:
        """Decode codeword to reconstructed point cloud."""
        device = next(self.parameters()).device
        sampling = torch.empty(0, device=device) if initial_sampling is None else initial_sampling.to(device)
        decoded = self.decoder(w, self.m, sampling)
        return Outputs(decoded.cloud.transpose(2, 1).contiguous(), decoded.attention, decoded.components)


class VQVAE(AE):
    """Vector-quantized point cloud autoencoder."""

    dim_codes: Final[int]
    book_size: Final[int]
    embedding_dim: Final[int]
    codebook: nn.Parameter

    def __init__(self) -> None:
        """Initialize."""
        super().__init__()
        cfg = Experiment.get_config().autoencoding
        self.dim_codes = cfg.model.dim_codes
        self.book_size = cfg.model.book_size
        self.embedding_dim = cfg.model.embedding_dim
        self.codebook = nn.Parameter(torch.randn(self.dim_codes, self.book_size, self.embedding_dim))
        return

    def quantise(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize continuous codeword chunks to nearest codebook embeddings."""
        batch, _ = x.size()
        book = self.codebook.repeat(batch, 1, 1)
        idx = self._lazy_square_distances(x.float(), book.float()).argmin(axis=2).view(-1, 1, 1)
        cw_embed = self.get_quantised_code(idx, book)
        one_hot_idx = torch.zeros(batch, self.dim_codes, self.book_size, device=x.device)
        one_hot_idx = one_hot_idx.scatter_(2, idx.view(batch, self.dim_codes, 1), 1)
        return cw_embed, one_hot_idx

    @override
    def decode(self, w_q: torch.Tensor, /, initial_sampling: torch.Tensor | None = None) -> Outputs:
        """Quantize and decode the unquantized codeword ``w_q``."""
        w_e, one_hot_idx = self.quantise(w_q)
        outputs = super().decode(TransferGrad.apply(w_e, w_q), initial_sampling)
        outputs.quantization = Quantization(w_q, w_e, one_hot_idx)
        return outputs

    def decode_nearest_codes(self, w_recon: torch.Tensor, /, initial_sampling: torch.Tensor | None = None) -> Outputs:
        """Decode the codes nearest to the codeword a prior reconstructed."""
        w_e, _ = self.quantise(w_recon)
        return super().decode(w_e, initial_sampling)

    def square_distances(self, x: torch.Tensor) -> torch.Tensor:
        """Squared distance of every chunk of a codeword from every code of its group.

        The codebook is detached: the distances choose codes, and do not train them.

        Args:
            x: codewords ``(B, dim_codes * embedding_dim)``.

        Returns:
            The distances ``(B, dim_codes, book_size)``.
        """
        batch, _ = x.size()
        book = self.codebook.detach().repeat(batch, 1, 1)
        distance = self._lazy_square_distances(x.float(), book.float())
        return distance.sum(1).view(batch, self.dim_codes, self.book_size)

    def get_quantised_code(self, idx: torch.Tensor, book: torch.Tensor) -> torch.Tensor:
        """Look up quantized vector embedding from codebook."""
        idx_exp = idx.expand(-1, -1, self.embedding_dim)
        return book.gather(1, idx_exp).view(-1, self.dim_codes * self.embedding_dim)

    def _lazy_square_distances(self, x: torch.Tensor, book: torch.Tensor) -> LazyTensor:
        batch, _ = x.size()
        x_chunks = x.view(batch * self.dim_codes, 1, self.embedding_dim)
        return pykeops_square_distance(x_chunks, book)


def get_autoencoder() -> BaseModel:
    """Instantiate the autoencoder model head specified in configuration."""
    cfg = Experiment.get_config().autoencoding.model
    model_map: dict[ModelHeads, type[BaseModel]] = {
        ModelHeads.AE: AE,
        ModelHeads.ORACLE: Oracle,
        ModelHeads.VQVAE: VQVAE,
    }
    return model_map[cfg.model_head]()
