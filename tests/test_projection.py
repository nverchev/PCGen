"""Tests for dimensionality reduction and projection utilities."""

import torch

from src.utils.projection import project_to_3d


def test_project_to_3d_high_dimension() -> None:
    """Project tensors with dimension > 3 to 3D via PCA."""
    latent_high = torch.randn(8, 16)
    pseudo_high = torch.randn(4, 16)
    l3, p3 = project_to_3d(latent_high, pseudo_high)

    assert l3.shape == (8, 3)
    assert p3.shape == (4, 3)
    return


def test_project_to_3d_three_dimensions() -> None:
    """Pass 3D tensors through unchanged."""
    latent_3d = torch.randn(5, 3)
    pseudo_3d = torch.randn(2, 3)
    l3_same, p3_same = project_to_3d(latent_3d, pseudo_3d)

    assert l3_same.shape == (5, 3)
    assert p3_same.shape == (2, 3)
    return
