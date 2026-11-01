"""Synthetic dataset of coins pierced by square holes."""

from typing import Final

import numpy as np
import torch

from numpy import typing as npt

from src.config import Experiment
from src.data.split import DataSplit
from src.data.structures import Inputs, Targets

__all__ = ['CoinsSplit']

HALF_THICKNESS: Final = 0.1
WIDTH: Final = 2 * HALF_THICKNESS
RADIUS: Final = float(np.sqrt(1 - HALF_THICKNESS**2))
N_COINS: Final = 4096 * 4
MAX_HOLES: Final = 3


class CoinsSplit(DataSplit):
    """Split generating a pierced coin on every access.

    The coins are built on demand and never stored, so the index selects
    nothing and every partition draws from the same distribution.

    Attributes:
        n_input_points: Points each coin holds.
    """

    n_input_points: Final[int]

    def __init__(self) -> None:
        """Initialize."""
        super().__init__()
        self.n_input_points = Experiment.get_config().data.dataset.input_points
        return

    def __getitem__(self, index: int) -> tuple[Inputs, Targets]:
        _not_used = index
        n_holes = np.random.randint(low=1, high=MAX_HOLES + 1)
        coin = self._create_coin().astype(np.float32)
        points: list[npt.NDArray[np.floating]] = []
        for _ in range(n_holes):
            while True:
                point = self._sample_point()
                if self._overlaps(point, points):
                    continue

                points.append(point)
                self._create_hole(coin, point)
                break

        cloud = torch.from_numpy(coin)
        return Inputs(cloud=cloud), Targets(ref_cloud=cloud)

    def __len__(self) -> int:
        return N_COINS

    def _create_coin(self) -> npt.NDArray[np.floating]:
        # the faces and the rim share the points in proportion to their area
        n_faces = round(self.n_input_points * 2 * RADIUS / (2 * RADIUS + 4 * HALF_THICKNESS))
        face_thetas = 2 * np.pi * np.random.rand(n_faces)
        face_radii = np.sqrt(np.random.rand(n_faces))
        face_z = np.random.choice([HALF_THICKNESS, -HALF_THICKNESS], size=n_faces, replace=True)
        n_rim = self.n_input_points - n_faces
        rim_thetas = 2 * np.pi * np.random.rand(n_rim)
        rim_z = (2 * np.random.rand(n_rim) - 1) * HALF_THICKNESS
        x = np.hstack([face_radii * np.cos(face_thetas), np.cos(rim_thetas)])
        y = np.hstack([face_radii * np.sin(face_thetas), np.sin(rim_thetas)])
        z = np.hstack([face_z, rim_z])
        return np.stack([x, y, z], axis=1)

    def _create_hole(self, coin: npt.NDArray[np.floating], point: npt.NDArray[np.floating]) -> None:
        # the points inside the square hole move onto the sides that close it
        hole_index = np.logical_and(abs(coin[:, 0] - point[0]) < WIDTH, abs(coin[:, 1] - point[1]) < WIDTH)
        hole = coin[hole_index]
        upper_index = hole[:, 2] == HALF_THICKNESS
        lower_index = np.logical_not(upper_index)
        upper = hole[upper_index]
        lower = hole[lower_index]
        tiled = np.tile(point.reshape(1, 3), [upper.shape[0], 1])
        upper_side = np.where(
            np.expand_dims(upper[:, 0] < tiled[:, 0], axis=1),
            np.stack([tiled[:, 0] - WIDTH, upper[:, 1], tiled[:, 0] - upper[:, 0] - HALF_THICKNESS], axis=1),
            np.stack([tiled[:, 0] + WIDTH, upper[:, 1], upper[:, 0] - tiled[:, 0] - HALF_THICKNESS], axis=1),
        )
        tiled = np.tile(point.reshape(1, 3), [lower.shape[0], 1])
        lower_side = np.where(
            np.expand_dims(lower[:, 1] < tiled[:, 1], axis=1),
            np.stack([lower[:, 0], tiled[:, 1] - WIDTH, tiled[:, 1] - lower[:, 1] - HALF_THICKNESS], axis=1),
            np.stack([lower[:, 0], WIDTH + tiled[:, 1], lower[:, 1] - tiled[:, 1] - HALF_THICKNESS], axis=1),
        )
        hole[upper_index] = upper_side
        hole[lower_index] = lower_side
        coin[hole_index] = hole
        return

    def _sample_point(self) -> npt.NDArray[np.floating]:
        # a center this far from the rim leaves room for the whole hole
        radius = RADIUS - np.sqrt(2) * WIDTH
        theta = 2 * np.pi * np.random.rand(1)
        z = np.random.choice([HALF_THICKNESS, -HALF_THICKNESS], 1)
        return np.hstack([radius * np.cos(theta), radius * np.sin(theta), z])

    @staticmethod
    def _overlaps(point: npt.NDArray[np.floating], past_points: list[npt.NDArray[np.floating]]) -> bool:
        return any(
            abs(past_point[0] - point[0]) < WIDTH or abs(past_point[1] - point[1]) < WIDTH for past_point in past_points
        )
