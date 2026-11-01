"""Normalization of raw point clouds."""

import numpy as np
import torch

from numpy import typing as npt

__all__ = ['normalize', 'normalize_clouds']


def normalize(cloud: npt.NDArray[np.floating], /) -> tuple[npt.NDArray[np.floating], float]:
    """Center a cloud and fit it inside the unit ball.

    Args:
        cloud: coordinates ``(n_points, 3)``, left unchanged.

    Returns:
        The centered cloud divided by its largest radius, and that radius.
    """
    centered = cloud - cloud.mean(axis=0)
    scale = float(np.max(np.sqrt(np.sum(centered**2, axis=1))))
    return centered / scale, scale


def normalize_clouds(clouds: torch.Tensor, /) -> torch.Tensor:
    """Center clouds and fit them inside the unit ball, as the dataset is.

    Args:
        clouds: clouds ``(B, M, 3)``, left unchanged.

    Returns:
        The normalized clouds.
    """
    centered = clouds - clouds.mean(dim=1, keepdim=True)
    scale = centered.norm(dim=2).max(dim=1)[0].view(-1, 1, 1)
    return centered / scale
