"""Data loaders of the two training phases."""

import torch

from torch.utils import data

from drytorch.lib import loading

from src.config import Experiment
from src.data import WDataset
from src.data.structures import Inputs, Targets

__all__ = [
    'get_autoencoder_loader',
    'get_w_encoder_loader',
]


def get_autoencoder_loader(
    split: data.Dataset[tuple[Inputs, Targets]], /
) -> loading.DataLoader[tuple[Inputs, Targets]]:
    """Get the loader of the autoencoder phase.

    Args:
        split: clouds to load.

    Returns:
        A loader drawing the configured batch size, with the configured
        number of worker processes.
    """
    cfg = Experiment.get_config()
    return loading.DataLoader(
        split,
        batch_size=cfg.autoencoding.train.batch_size,
        n_workers=cfg.user.n_workers,
    )


def get_w_encoder_loader(split: WDataset, /) -> loading.DataLoader[tuple[torch.Tensor, torch.Tensor]]:
    """Get the loader of the codeword phase.

    The codewords come out of the autoencoder already on the device, so they
    are not pinned and not loaded in another process.

    Args:
        split: codewords to load.

    Returns:
        A loader drawing the configured batch size.
    """
    cfg_w = Experiment.get_config().w_autoencoding
    assert cfg_w is not None
    return loading.DataLoader(
        split,
        batch_size=cfg_w.train.batch_size,
        pin_memory=False,
    )
