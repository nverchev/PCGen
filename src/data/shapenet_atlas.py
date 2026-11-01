"""Builder of the ShapeNet clouds AtlasNet distributes as meshes."""

import logging
import pathlib

from typing import ClassVar, Final, override

import numpy as np
import tqdm

from src.data.databuilder import DataBuilder
from src.data.split import CloudSplit, Partitions

__all__ = ['ShapeNetAtlasDataBuilder']

SOURCE: Final = 'https://github.com/TheoDEPRELLE/AtlasNetV2'
CAST_MARKER: Final = '.float32'


class ShapeNetAtlasDataBuilder(DataBuilder):
    """Clouds read from the AtlasNet meshes, one file each.

    A mesh gives as many points as it holds, so the clouds stay separate files
    and are handed over unnormalized, for the split to normalize each sampling.
    """

    folder_processed: ClassVar[str] = 'shapenet'
    folder_raw: ClassVar[str] = 'customShapeNet'
    metadata_file: ClassVar[str] = 'shapenet_AtlasNet_classes.json'

    test_ratio: ClassVar[float] = 0.2
    val_ratio: ClassVar[float] = 0.2

    _paths: dict[Partitions, list[pathlib.Path]]

    @override
    def get_split(self, partition: Partitions) -> CloudSplit:
        # the clouds are memory mapped, the whole dataset being too large to hold
        return CloudSplit([np.load(path, mmap_mode='r') for path in self._paths[partition]])

    @override
    def _prepare(self) -> None:
        if not self.dir_raw.exists():
            msg = f'Download {self.folder_raw} to {self.dir_raw} as described in {SOURCE}.'
            raise ValueError(msg)

        for class_name, code in self.classes.items():
            class_dir = self.dir_processed / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            ply_paths = sorted((self.dir_raw / code / 'ply').glob('*.ply'))
            for index, ply_path in enumerate(tqdm.tqdm(ply_paths, desc=class_name)):
                _convert_ply(ply_path, class_dir / str(index))

            _cast_prepared(class_dir, desc=class_name)

        self._paths = {partition: self._get_paths(partition) for partition in Partitions}
        return

    def _get_paths(self, partition: Partitions) -> list[pathlib.Path]:
        paths: list[pathlib.Path] = []
        for class_name in self.classes:
            class_paths = sorted((self.dir_processed / class_name).glob('*.npy'))
            first_split = int(len(class_paths) * (1 - self.val_ratio - self.test_ratio))
            second_split = int(len(class_paths) * (1 - self.test_ratio))
            if partition is Partitions.train:
                paths.extend(class_paths[:first_split])
            elif partition is Partitions.train_val:
                paths.extend(class_paths[:second_split])
            elif partition is Partitions.val:
                paths.extend(class_paths[first_split:second_split])
            else:
                paths.extend(class_paths[second_split:])

        return paths


def _convert_ply(ply_path: pathlib.Path, npy_path: pathlib.Path) -> None:
    # a file name holding '*.' is corrupted rather than a pattern
    if '*.' in ply_path.name or npy_path.with_suffix('.npy').exists():
        return

    try:
        cloud = np.loadtxt(ply_path, skiprows=12, usecols=(0, 1, 2))
    except UnicodeDecodeError:
        logging.warning(f'{ply_path} is corrupted.')
    else:
        np.save(npy_path, cloud.astype(np.float32))

    return


def _cast_prepared(class_dir: pathlib.Path, *, desc: str) -> None:
    # an earlier version prepared the clouds as float64, doubling every read
    marker = class_dir / CAST_MARKER
    if marker.exists():
        return

    for path in tqdm.tqdm(sorted(class_dir.glob('*.npy')), desc=f'{desc} to float32'):
        cloud = np.load(path)
        if cloud.dtype != np.float32:
            np.save(path, cloud.astype(np.float32))

    marker.touch()
    return
