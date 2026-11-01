"""Storage of the prepared point clouds."""

import dataclasses
import enum
import pathlib

import h5py
import numpy as np

from numpy import typing as npt

__all__ = [
    'Clouds',
    'H5Keys',
    'concat_clouds',
    'load_clouds',
    'save_clouds',
]


class H5Keys(enum.StrEnum):
    """Dataset keys of a preprocessed file."""

    CLOUDS = 'clouds'
    SCALES = 'scales'


@dataclasses.dataclass(frozen=True, slots=True)
class Clouds:
    """Normalized point clouds and the scales that restore them.

    Attributes:
        clouds: Normalized coordinates ``(n_clouds, n_points, 3)``.
        scales: Factor each cloud was divided by ``(n_clouds,)``.
    """

    clouds: npt.NDArray[np.float32]
    scales: npt.NDArray[np.float32]


def concat_clouds(parts: list[Clouds], /) -> Clouds:
    """Join several collections into one.

    Args:
        parts: collections to join, in the order they are given.

    Returns:
        A collection holding every cloud of every part.

    Raises:
        ValueError: if no part is given.
    """
    if not parts:
        raise ValueError('No clouds to join.')

    return Clouds(
        clouds=np.concatenate([part.clouds for part in parts], axis=0),
        scales=np.concatenate([part.scales for part in parts], axis=0),
    )


def load_clouds(path: pathlib.Path, /) -> Clouds:
    """Read a preprocessed file.

    Args:
        path: file written by :func:`save_clouds`.

    Returns:
        The clouds and scales the file holds.

    Raises:
        FileNotFoundError: if the file does not exist.
        KeyError: if the file misses a dataset, having been written by an
            earlier version of the preprocessing.
    """
    if not path.exists():
        msg = f'No preprocessed file at {path}.'
        raise FileNotFoundError(msg)

    with h5py.File(path, 'r', locking=False) as file:
        missing = [key for key in H5Keys if key not in file]
        if missing:
            msg = f'{path} misses {missing}: delete it and preprocess again.'
            raise KeyError(msg)

        clouds = _read(file, H5Keys.CLOUDS)
        scales = _read(file, H5Keys.SCALES)

    return Clouds(clouds=clouds, scales=scales)


def save_clouds(path: pathlib.Path, clouds: Clouds, /) -> None:
    """Write a preprocessed file, keeping an existing one.

    Args:
        path: file to write, whose parent is created when missing.
        clouds: collection to store.

    Side Effects:
        Creates the parent directory and the file.
    """
    if path.exists():
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, 'w') as file:
        file.create_dataset(H5Keys.CLOUDS, data=clouds.clouds)
        file.create_dataset(H5Keys.SCALES, data=clouds.scales)

    return


def _read(file: h5py.File, key: H5Keys) -> npt.NDArray[np.float32]:
    dataset = file[key]
    if not isinstance(dataset, h5py.Dataset):
        msg = f'{key} is not a dataset in {file.filename}.'
        raise KeyError(msg)

    return dataset[()].astype(np.float32)
