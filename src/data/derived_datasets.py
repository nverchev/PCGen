"""Dataset of the codewords a trained autoencoder produces."""

from __future__ import annotations

import contextlib

from collections.abc import Iterator, Sized
from typing import TYPE_CHECKING, Final

import torch
import torch.nn as nn
from torch.utils import data

from src.data.structures import Inputs, Targets

if TYPE_CHECKING:
    from src.lib.models import VQVAEModel

__all__ = [
    'WDataset',
]

_BATCH_NORMS: Final = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)


class WDataset(data.Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Codewords of a split, with the codes the autoencoder chose for them.

    The codeword phase trains on what the first phase encodes, so the clouds go
    through the encoder once per access and only the codewords reach the model.

    Attributes:
        device: Device on which the autoencoder model resides.
        model: Autoencoder model encoding the clouds.
        n_clouds: Number of clouds the split holds.
        split: Clouds to encode.
    """

    device: Final[torch.device]
    model: Final[VQVAEModel]
    n_clouds: Final[int]
    split: Final[data.Dataset[tuple[Inputs, Targets]]]

    def __init__(
        self,
        split: data.Dataset[tuple[Inputs, Targets]],
        model: VQVAEModel,
        /,
    ) -> None:
        """Initialize.

        Args:
            split: clouds to encode.
            model: autoencoder model encoding them.

        Raises:
            TypeError: if the split does not state how many clouds it holds.
        """
        super().__init__()
        if not isinstance(split, Sized):
            msg = 'The codewords need a split of known length.'
            raise TypeError(msg)

        self.model = model
        self.device = model.device
        self.n_clouds = len(split)
        self.split = split
        return

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.__getitems__([index])[0]

    def __getitems__(self, indices: list[int]) -> list[tuple[torch.Tensor, torch.Tensor]]:
        self.model.module.train(not torch.is_inference_mode_enabled())
        clouds = torch.stack([self.split[index][0].cloud for index in indices]).to(self.device)
        return self._encode(clouds)

    def __len__(self) -> int:
        return self.n_clouds

    @torch.inference_mode()
    def _encode(self, clouds: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor]]:
        with _batch_statistics_only(self.model.module):
            w_q = self.model.encode(clouds)

        one_hot_idx = self.model.module.quantise(w_q)[1]
        return list(zip(w_q.unbind(), one_hot_idx.unbind(), strict=True))


@contextlib.contextmanager
def _batch_statistics_only(module: nn.Module, /) -> Iterator[None]:
    # in training mode, batch normalization then uses the statistics of the batch without updating the running ones
    norms = [layer for layer in module.modules() if isinstance(layer, _BATCH_NORMS) and layer.track_running_stats]
    for layer in norms:
        layer.track_running_stats = False

    try:
        yield
    finally:
        for layer in norms:
            layer.track_running_stats = True

    return
