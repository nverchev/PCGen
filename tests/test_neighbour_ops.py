"""Tests for the operations on point neighbourhoods."""

import torch

from src.utils.neighbour_ops import graph_filtering


def test_graph_filtering_has_a_finite_gradient_when_points_coincide() -> None:
    """Two points in the same place, which the decoder can produce, do not make the gradient non-finite."""
    torch.manual_seed(0)
    cloud = torch.rand(1, 3, 64)
    cloud[..., 1] = cloud[..., 0]
    cloud.requires_grad_(True)

    graph_filtering(cloud).sum().backward()

    assert cloud.grad is not None
    assert torch.isfinite(cloud.grad).all()
    return
