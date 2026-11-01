"""Tests for the distances between point clouds."""

from collections.abc import Callable

import pytest
import torch

from src.utils.distances import chamfer, emd, square_distance, torch_chamfer, torch_emd

CLOUD = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])


def test_square_distance_holds_every_pair() -> None:
    """The entries are the squared distances of the point pairs."""
    other = torch.tensor([[[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]]])

    distances = square_distance(CLOUD, other)

    expected = torch.tensor([[[0.0, 4.0], [1.0, 5.0], [1.0, 5.0]]])
    torch.testing.assert_close(distances, expected)


def test_chamfer_vanishes_on_a_repeated_cloud() -> None:
    """A cloud has no Chamfer distance to itself."""
    assert float(torch_chamfer(CLOUD, CLOUD.clone())) == pytest.approx(0.0)


def test_chamfer_ignores_the_order_of_the_points() -> None:
    """Permuting the points leaves the Chamfer distance unchanged."""
    other = torch.tensor([[[0.0, 0.0, 1.0], [1.0, 1.0, 1.0], [2.0, 0.0, 0.0]]])

    straight = torch_chamfer(CLOUD, other)
    permuted = torch_chamfer(CLOUD, other[:, [2, 0, 1]])

    torch.testing.assert_close(straight, permuted)


def test_chamfer_sums_both_directions() -> None:
    """The distance adds the matches of each cloud to the other."""
    first = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]])
    second = torch.tensor([[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]])

    distance = torch_chamfer(first, second)

    # the far points match each other, contributing one in each direction
    torch.testing.assert_close(distance, torch.tensor([2.0]))


def test_emd_matches_every_point_once() -> None:
    """The optimal matching pairs the points one to one."""
    first = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]])
    second = torch.tensor([[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]])

    distance = torch_emd(first, second)

    torch.testing.assert_close(distance, torch.tensor([1.0]))


def test_emd_sums_euclidean_distances() -> None:
    """The matched points add their distance, not its square."""
    first = torch.tensor([[[0.0, 0.0, 0.0]]])
    second = torch.tensor([[[3.0, 4.0, 0.0]]])

    distance = torch_emd(first, second)

    torch.testing.assert_close(distance, torch.tensor([5.0]))


def test_emd_has_a_finite_gradient_for_coinciding_points() -> None:
    """Points that coincide do not make the gradient infinite."""
    first = CLOUD.clone().requires_grad_()

    torch_emd(first, CLOUD).sum().backward()

    assert first.grad is not None
    assert torch.isfinite(first.grad).all()


def test_emd_ignores_the_order_of_the_points() -> None:
    """Permuting the points leaves the matching cost unchanged."""
    other = torch.tensor([[[0.0, 0.0, 1.0], [1.0, 1.0, 1.0], [2.0, 0.0, 0.0]]])

    straight = torch_emd(CLOUD, other)
    permuted = torch_emd(CLOUD, other[:, [2, 0, 1]])

    torch.testing.assert_close(straight, permuted)


def test_emd_rejects_clouds_of_different_size() -> None:
    """Clouds of different size have no one to one matching."""
    with pytest.raises(ValueError, match='equal size'):
        torch_emd(CLOUD, CLOUD[:, :2])


@pytest.mark.parametrize('distance_fn', [square_distance, torch_chamfer, chamfer, torch_emd, emd])
def test_distance_functions_return_float32_for_float16_inputs(
    distance_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> None:
    """Every distance function converts float16 inputs to float32."""
    c1 = CLOUD.half()
    c2 = CLOUD.half()

    result = distance_fn(c1, c2)

    assert result.dtype == torch.float32
