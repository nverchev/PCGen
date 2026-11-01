"""Distances between point clouds.

The Chamfer distance sums the squared distances of the matched points, as in
the original implementation, so a caller comparing clouds of different sizes
divides by the number of points itself.
"""

import numpy as np
import torch

from scipy import optimize

__all__ = [
    'chamfer',
    'emd',
    'square_distance',
    'torch_chamfer',
    'torch_emd',
]


def square_distance(t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
    """Return the squared distance of every pair of points of two clouds.

    Args:
        t1: first cloud ``(B, N, D)``.
        t2: second cloud ``(B, M, D)``.

    Returns:
        Squared distances ``(B, N, M)``.
    """
    t1, t2 = t1.float(), t2.float()
    transposed = t2.transpose(-1, -2)
    distance = -2 * torch.matmul(t1, transposed)
    distance += torch.sum(t1**2, -1, keepdim=True)
    distance += torch.sum(transposed**2, -2, keepdim=True)
    return distance


def torch_chamfer(t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
    """Return the Chamfer distance of two clouds, summed over the points.

    Args:
        t1: first cloud ``(B, N, D)``.
        t2: second cloud ``(B, M, D)``.

    Returns:
        Summed squared distances of the matched points ``(B,)``.
    """
    t1, t2 = t1.float(), t2.float()
    distance = square_distance(t1, t2)
    return torch.min(distance, dim=-1)[0].sum(1) + torch.min(distance, dim=-2)[0].sum(1)


def chamfer(t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
    """Return the Chamfer distance, with the keops backend when it is usable.

    Args:
        t1: first cloud ``(B, N, D)``.
        t2: second cloud ``(B, M, D)``.

    Returns:
        Summed squared distances of the matched points ``(B,)``.
    """
    t1, t2 = t1.float(), t2.float()
    if t1.device.type != 'cuda':
        return torch_chamfer(t1, t2)

    try:
        from pykeops.torch import LazyTensor
    except (ImportError, ModuleNotFoundError):
        return torch_chamfer(t1, t2)

    lazy1 = LazyTensor(t1[:, :, None, :])
    lazy2 = LazyTensor(t2[:, None, :, :])
    distance = ((lazy1 - lazy2) ** 2).sum(-1)
    matched1 = t1.gather(1, distance.argmin(axis=1).expand(-1, -1, t1.shape[2]))
    matched2 = t2.gather(1, distance.argmin(axis=2).expand(-1, -1, t2.shape[2]))
    return ((t2 - matched1) ** 2).sum((1, 2)) + ((t1 - matched2) ** 2).sum((1, 2))


def torch_emd(t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
    """Return the exact earth mover's distance of two clouds of equal size.

    Args:
        t1: first cloud ``(B, N, D)``.
        t2: second cloud ``(B, N, D)``.

    Returns:
        Summed squared distances of the optimal matching ``(B,)``.

    Raises:
        ValueError: if the two clouds hold a different number of points.
    """
    t1, t2 = t1.float(), t2.float()
    if t1.shape[1] != t2.shape[1]:
        msg = 'The earth mover distance needs clouds of equal size.'
        raise ValueError(msg)

    distance = torch.sqrt(torch.clamp(square_distance(t1, t2), min=1e-12))
    as_numpy = distance.detach().cpu().numpy()
    costs = torch.zeros(t1.shape[0], device=t1.device)
    rows = torch.arange(t1.shape[1])
    for sample in range(t1.shape[0]):
        _not_used, columns = optimize.linear_sum_assignment(as_numpy[sample])
        costs[sample] = distance[sample, rows, np.asarray(columns)].sum()

    return costs


def emd(t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
    """Return the earth mover's distance, with the matching extension when built.

    Args:
        t1: first cloud ``(B, N, D)``.
        t2: second cloud ``(B, N, D)``.

    Returns:
        Summed squared distances of the matching ``(B,)``.
    """
    t1, t2 = t1.float(), t2.float()
    if t1.device.type != 'cuda':
        return torch_emd(t1, t2)

    try:
        from structural_losses import match_cost
    except (ImportError, ModuleNotFoundError):
        return torch_emd(t1, t2)

    return match_cost(t1.contiguous(), t2.contiguous())
