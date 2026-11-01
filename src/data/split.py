"""Partitions of a dataset and the split feeding the model."""

from __future__ import annotations

import abc
import enum

from collections.abc import Sequence
from typing import Final

import numpy as np
import torch

from numpy import typing as npt
from torch.utils import data

from src.config import Experiment
from src.data.preprocessing import normalize
from src.data.structures import Inputs, Targets

__all__ = [
    'CloudSplit',
    'DataSplit',
    'Partitions',
]


class Partitions(enum.Enum):
    """Named dataset splits, ``train_val`` joining ``train`` and ``val``."""

    train = enum.auto()
    train_val = enum.auto()
    val = enum.auto()
    test = enum.auto()


class DataSplit(data.Dataset[tuple[Inputs, Targets]], abc.ABC):
    """Abstract base class for point cloud dataset splits."""

    @abc.abstractmethod
    def __getitem__(self, index: int) -> tuple[Inputs, Targets]:
        """Return the inputs and targets for a sample."""

    @abc.abstractmethod
    def __len__(self) -> int:
        """Return the number of samples in the split."""


class CloudSplit(DataSplit):
    """The clouds of one partition, sampled anew on every access.

    An epoch never sees the same points twice. Clouds given without their
    scales are not normalized yet, and every sampling of them is normalized on
    its own.

    Attributes:
        clouds: Clouds to sample, stacked or one entry each.
        n_input_points: Points each sampling holds.
        resample: Whether the reference cloud is a second sampling.
        scales: Factor each cloud was divided by, or ``None`` when the clouds
            are not normalized yet.
    """

    clouds: Final[Sequence[npt.NDArray[np.floating]] | npt.NDArray[np.floating]]
    n_input_points: Final[int]
    resample: Final[bool]
    scales: Final[npt.NDArray[np.floating] | None]

    def __init__(
        self,
        clouds: Sequence[npt.NDArray[np.floating]] | npt.NDArray[np.floating],
        /,
        scales: npt.NDArray[np.floating] | None = None,
    ) -> None:
        """Initialize.

        Args:
            clouds: clouds of one partition, stacked or one entry each.
            scales: factor each cloud was divided by, left out when the clouds
                are not normalized yet.
        """
        super().__init__()
        cfg_dataset = Experiment.get_config().data.dataset
        self.clouds = clouds
        self.n_input_points = cfg_dataset.input_points
        self.resample = cfg_dataset.resample
        self.scales = scales
        return

    def __getitem__(self, index: int) -> tuple[Inputs, Targets]:
        cloud, ref_cloud = self._draw_samplings(self.clouds[index])
        if self.scales is None:
            cloud, scale = normalize(cloud)
            ref_cloud = normalize(ref_cloud)[0]
        else:
            scale = float(self.scales[index])

        inputs = Inputs(cloud=torch.from_numpy(cloud))
        targets = Targets(ref_cloud=torch.from_numpy(ref_cloud), scale=torch.tensor(scale))
        return inputs, targets

    def __len__(self) -> int:
        return len(self.clouds)

    def _draw_samplings(
        self, cloud: npt.NDArray[np.floating]
    ) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
        pool = np.arange(cloud.shape[0])
        if not self.resample:
            drawn = np.asarray(cloud[np.random.choice(pool, size=self.n_input_points, replace=False)], np.float32)
            return drawn, drawn

        # a cloud too small for two disjoint samplings has to draw points twice
        replace = 2 * self.n_input_points > cloud.shape[0]
        sampling = np.random.choice(pool, size=2 * self.n_input_points, replace=replace)
        drawn = np.asarray(cloud[sampling], np.float32)
        return drawn[: self.n_input_points], drawn[self.n_input_points :]
