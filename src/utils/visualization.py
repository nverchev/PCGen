"""Visualization helpers for running inference and publishing events."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from src.lib.events import GenerationEvent, ReconstructionEvent
from src.data.structures import Inputs

if TYPE_CHECKING:
    from src.module import BaseModel, VQVAEWithVAE

__all__ = [
    'generate_and_publish',
    'reconstruct_and_publish',
]


def reconstruct_and_publish(
    module: BaseModel | VQVAEWithVAE,
    clouds: torch.Tensor,
    /,
    *,
    sample_indices: Sequence[int],
    epoch: int,
    loop: bool,
    is_double: bool | None = None,
    source_name: str | None = None,
) -> None:
    """Reconstruct point clouds and publish a ReconstructionEvent.

    Args:
        module: Model reconstructing the clouds.
        clouds: Point clouds to reconstruct ``(B, N, 3)``.
        sample_indices: Sample indices corresponding to the batch.
        epoch: Current training epoch.
        loop: whether the sampling ends with a closed path through the corners of the sampling space.
        is_double: Whether this is a double reconstruction (defaults to checking module type).
        source_name: Optional name of the model or evaluation source.
    """
    device = next(module.parameters()).device
    with torch.inference_mode():
        module.eval()
        sampling = module.get_sampling(clouds.shape[0], loop=loop)
        inputs = Inputs(cloud=clouds.to(device), initial_sampling=sampling)
        outputs = module(inputs)

    if is_double is None:
        from src.module.double_encoders import VQVAEWithVAE

        is_double = isinstance(module, VQVAEWithVAE)

    default_source = 'double_reconstruction' if is_double else 'reconstruction'
    ReconstructionEvent(
        outputs=outputs,
        sample_indices=sample_indices,
        epoch=epoch,
        inputs=inputs,
        is_double=is_double,
        source_name=source_name if source_name is not None else default_source,
    )
    return


def generate_and_publish(
    module: VQVAEWithVAE,
    /,
    *,
    n_clouds: int,
    z_bias: torch.Tensor,
    epoch: int,
    loop: bool,
) -> None:
    """Generate point clouds from prior and publish a GenerationEvent.

    Args:
        module: autoencoder and prior to sample from.
        n_clouds: Number of clouds to sample.
        z_bias: Latent bias tensor.
        epoch: Current epoch number.
        loop: whether the sampling ends with a closed path through the corners of the sampling space.
    """
    with torch.inference_mode():
        outputs = module.random_sampling(n_clouds, module.get_sampling(n_clouds, loop=loop), z_bias)

    GenerationEvent(outputs=outputs, epoch=epoch)
    return
