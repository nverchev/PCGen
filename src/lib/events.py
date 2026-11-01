"""Events published during training and evaluation."""

from __future__ import annotations

import dataclasses

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from drytorch.core import log_events

if TYPE_CHECKING:
    from src.data.structures import Inputs, Outputs

__all__ = [
    'GenerationEvent',
    'LatentSpaceEvent',
    'ReconstructionEvent',
]


@dataclasses.dataclass(frozen=True)
class LatentSpaceEvent(log_events.Event):
    """Event logged when the latent space is evaluated.

    Attributes:
        source_name: Name of the runner that evaluated the model.
        model_name: Name of the evaluated model.
        epoch: Epoch of the evaluated weights.
        latent: Continuous latent points tensor projected to 3D ``(N, 3)``.
        pseudo_latent: Pseudo-input latent points tensor projected to 3D ``(M, 3)``.
    """

    source_name: str
    model_name: str
    epoch: int
    latent: torch.Tensor
    pseudo_latent: torch.Tensor


@dataclasses.dataclass(frozen=True)
class ReconstructionEvent(log_events.Event):
    """Event logged when point clouds are reconstructed.

    Attributes:
        outputs: Reconstruction model outputs containing reconstructed clouds,
            attention, components, etc.
        sample_indices: Sequence of indices for the reconstructed samples.
        epoch: Current epoch number.
        inputs: Optional model inputs containing original clouds.
        is_double: Whether this is a double reconstruction through the codeword prior.
        source_name: Name of the runner/source emitting the reconstructions.
        model_name: Name of the evaluated model.
    """

    outputs: Outputs
    sample_indices: Sequence[int]
    epoch: int
    inputs: Inputs | None = None
    is_double: bool = False
    source_name: str = 'reconstruction'
    model_name: str = 'autoencoder'


@dataclasses.dataclass(frozen=True)
class GenerationEvent(log_events.Event):
    """Event logged when point clouds are generated from prior sampling.

    Attributes:
        outputs: Generation model outputs containing sampled point clouds.
        epoch: Current epoch number.
        source_name: Name of the runner/source emitting the generations.
        model_name: Name of the generator model.
    """

    outputs: Outputs
    epoch: int
    source_name: str = 'generation'
    model_name: str = 'autoencoder'
