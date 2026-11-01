"""Scores comparing a generated set of clouds with a reference set.

The three scores follow the definitions used in the point cloud generation
literature, over the distances between the reference and the generated clouds:

* the minimum matching distance (MMD) averages, over the reference clouds, the
  distance to the closest generated one;
* the coverage is the fraction of reference clouds that are the closest to at
  least one generated cloud;
* the one nearest neighbour accuracy (1-NNA) is the fraction of all clouds whose
  closest other cloud belongs to their own set, so a value near one half means
  the two sets are indistinguishable.

The scores need all the generated clouds at once, so they are not a drytorch
objective: they run on clouds already generated, and they publish their values
with the same event as the runners.
"""

from __future__ import annotations

import logging

from collections.abc import Callable, Sequence
from typing import Any, Final

import torch

from drytorch.core import log_events
from drytorch.core import protocols as p

__all__ = [
    'GenerativeScores',
    'get_generative_scores',
    'get_pairwise_distances',
]

BLOCK_SIZE: Final = 32


class GenerativeScores:
    """Scores of generated clouds against a reference set.

    The distances inside the reference set are measured once, when the scores
    are created.

    Attributes:
        reference: the reference clouds, each ``(N, 3)``.
        distance: distance between two batches of clouds, summed over points.
        name: name under which the scores are published.
        device: where to measure the distances.
    """

    reference: Final[Sequence[torch.Tensor]]
    distance: Final[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]
    name: Final[str]
    device: Final[torch.device]
    _reference_distances: Final[torch.Tensor]

    def __init__(
        self,
        reference: Sequence[torch.Tensor],
        distance: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        /,
        *,
        name: str = 'Generation',
        device: torch.device,
    ) -> None:
        """Initialize.

        Args:
            reference: the reference clouds, each ``(N, 3)``.
            distance: distance between two batches of clouds, summed over points.
            name: name under which the scores are published.
            device: where to measure the distances.
        """
        self.reference = reference
        self.distance = distance
        self.name = name
        self.device = device
        logging.info('Measuring the distances between the %d reference clouds for %s.', len(reference), name)
        self._reference_distances = get_pairwise_distances(reference, reference, distance, device=device)
        return

    def __call__(self, model: p.ModelProtocol[Any, Any], generated: Sequence[torch.Tensor]) -> dict[str, float]:
        """Get the scores of the clouds that a model generated, and publish them.

        Args:
            model: the model that generated the clouds.
            generated: the generated clouds, each ``(M, 3)``.

        Returns:
            The scores.

        Side Effects:
            Publishes the scores as metrics of the model at its current epoch.
        """
        logging.info('Calculating %s for %s.', self.name, model.name)
        cross_distances = get_pairwise_distances(self.reference, generated, self.distance, device=self.device)
        generated_distances = get_pairwise_distances(generated, generated, self.distance, device=self.device)
        scores = get_generative_scores(self._reference_distances, cross_distances, generated_distances)
        log_events.MetricEvent(model_name=model.name, source_name=self.name, epoch=model.epoch, metrics=scores)
        return scores


def get_generative_scores(
    reference_distances: torch.Tensor, cross_distances: torch.Tensor, generated_distances: torch.Tensor
) -> dict[str, float]:
    """Get the scores from the distances between and within the two sets.

    Args:
        reference_distances: distances between the reference clouds ``(R, R)``.
        cross_distances: distances from the reference to the generated clouds ``(R, G)``.
        generated_distances: distances between the generated clouds ``(G, G)``.

    Returns:
        The minimum matching distance (``MMD``), the coverage (``Coverage``)
        and the one nearest neighbour accuracy (``1-NNA``).

    Raises:
        ValueError: if a set has fewer than two clouds.
    """
    n_reference, n_generated = cross_distances.shape
    if n_reference < 2 or n_generated < 2:
        raise ValueError('The scores need at least two clouds in each set.')

    mmd = cross_distances.min(dim=1).values.mean()
    covered = cross_distances.argmin(dim=0).unique()
    upper = torch.cat([reference_distances, cross_distances], dim=1)
    lower = torch.cat([cross_distances.T, generated_distances], dim=1)
    all_distances = torch.cat([upper, lower], dim=0)

    # a cloud is not its own neighbour
    all_distances.fill_diagonal_(float('inf'))
    is_reference = torch.arange(n_reference + n_generated) < n_reference
    nearest = all_distances.argmin(dim=1)
    same_set = is_reference[nearest] == is_reference
    return {
        'MMD': mmd.item(),
        'Coverage': len(covered) / n_reference,
        '1-NNA': same_set.float().mean().item(),
    }


def get_pairwise_distances(
    first: Sequence[torch.Tensor],
    second: Sequence[torch.Tensor],
    distance: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    *,
    device: torch.device,
) -> torch.Tensor:
    """Get the distances, averaged over points, between every pair of clouds.

    Args:
        first: clouds of the rows, each ``(N, 3)``.
        second: clouds of the columns, each ``(N, 3)``.
        distance: distance between two batches of clouds, summed over points.
        device: where to measure the distances.

    Returns:
        The distances ``(len(first), len(second))``, on the CPU.
    """
    block = torch.zeros(len(first), len(second))
    if not len(first) or not len(second):
        return block

    # a whole row at once asks for one distance matrix per pair, so it goes in blocks
    n_points = second[0].shape[0]
    for row, cloud in enumerate(first):
        for start in range(0, len(second), BLOCK_SIZE):
            chunk = torch.stack(list(second[start : start + BLOCK_SIZE])).to(device).contiguous()
            repeated = cloud.unsqueeze(0).expand_as(chunk).to(device).contiguous()
            values = distance(chunk, repeated) / n_points
            block[row, start : start + chunk.shape[0]] = values.detach().cpu()

    return block
