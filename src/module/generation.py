"""Modules drawing clouds, from a trained prior or from a real set."""

from typing import override

import torch

from torch import nn

from src.data.preprocessing import normalize_clouds
from src.data.structures import Outputs
from src.module import double_encoders

__all__ = [
    'Drawing',
    'Generation',
    'Resampling',
]


class Drawing(nn.Module):
    """Wrapper exposing random sampling from the prior as forward."""

    module: double_encoders.VQVAEWithVAE

    def __init__(self, module: double_encoders.VQVAEWithVAE, /) -> None:
        """Initialize.

        Args:
            module: autoencoder and prior drawing clouds.
        """
        super().__init__()
        self.module = module
        return

    @override
    def forward(self, batch_size: int, initial_sampling: torch.Tensor | None = None) -> Outputs:
        """Draw a batch of clouds from the prior, on the given initial points if any."""
        return self.module.random_sampling(batch_size, initial_sampling)


class Generation(nn.Module):
    """Wrapper drawing as many clouds from the prior as it is given indices."""

    module: double_encoders.VQVAEWithVAE

    def __init__(self, module: double_encoders.VQVAEWithVAE, /) -> None:
        """Initialize.

        Args:
            module: autoencoder and prior drawing clouds.
        """
        super().__init__()
        self.module = module
        return

    @override
    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        """Draw one cloud per index, normalized the way the dataset is."""
        return normalize_clouds(self.module.random_sampling(indices.shape[0]).recon)


class Resampling(nn.Module):
    """Oracle drawing clouds from a real set, with replacement.

    Attributes:
        clouds: the clouds to draw from ``(C, N, 3)``.
    """

    clouds: torch.Tensor

    def __init__(self, clouds: torch.Tensor, /) -> None:
        """Initialize.

        Args:
            clouds: the clouds to draw from ``(C, N, 3)``.
        """
        super().__init__()
        self.register_buffer('clouds', clouds)
        return

    @override
    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        """Draw one cloud per index."""
        drawn = torch.randint(self.clouds.shape[0], (indices.shape[0],), device=self.clouds.device)
        return self.clouds[drawn]
