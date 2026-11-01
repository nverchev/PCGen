"""Builder of the ShapeNet clouds PointFlow distributes as fifteen thousand points."""

import logging
import pathlib

from typing import ClassVar, override

import numpy as np
import tqdm

from numpy import typing as npt

from src.data.databuilder import DataBuilder
from src.data.preprocessing import normalize
from src.data.split import CloudSplit, Partitions
from src.data.storage import Clouds, concat_clouds, load_clouds, save_clouds
from src.utils.download import download_extract_zip

__all__ = ['ShapeNetFlowDataBuilder']


class ShapeNetFlowDataBuilder(DataBuilder):
    """Clouds of a fixed size, normalized once and stored with their scale.

    Every cloud holds far more points than a sampling needs, so preprocessing
    normalizes it once and stores the scale beside it.
    """

    folder_processed: ClassVar[str] = 'shapenet_flow'
    folder_raw: ClassVar[str] = 'ShapeNetCore.v2.PC15k'
    metadata_file: ClassVar[str] = 'shapenet_PointFlow_classes.json'

    url: ClassVar[str] = 'https://gaimfs.ugent.be/Public/Dataset/ShapeNetCore.v2.PC15k.zip'

    _clouds: dict[Partitions, Clouds]

    @override
    def get_split(self, partition: Partitions) -> CloudSplit:
        clouds = self._get_clouds(partition)
        return CloudSplit(clouds.clouds, clouds.scales)

    @override
    def _prepare(self) -> None:
        download_extract_zip(self.dir_raw, self.url)
        for class_name, code in self.classes.items():
            for partition in (Partitions.train, Partitions.val, Partitions.test):
                h5_path = self.dir_processed / class_name / f'{partition.name}.h5'
                if h5_path.exists():
                    continue

                logging.info(f'Preparing the {partition.name} clouds of {class_name}.')
                raw_dir = self.dir_raw / code / partition.name
                save_clouds(h5_path, _read_clouds(raw_dir, desc=f'{class_name} {partition.name}'))

        self._clouds = {}
        return

    def _get_clouds(self, partition: Partitions) -> Clouds:
        if partition not in self._clouds:
            if partition is Partitions.train_val:
                stored = (Partitions.train, Partitions.val)
            else:
                stored = (partition,)

            parts = [self._read(class_name, one) for one in stored for class_name in self.classes]
            self._clouds[partition] = concat_clouds(parts)

        return self._clouds[partition]

    def _read(self, class_name: str, partition: Partitions) -> Clouds:
        return load_clouds(self.dir_processed / class_name / f'{partition.name}.h5')


def _read_clouds(raw_dir: pathlib.Path, *, desc: str) -> Clouds:
    paths = sorted(raw_dir.glob('*.npy'))
    if not paths:
        msg = f'No clouds in {raw_dir}.'
        raise ValueError(msg)

    clouds: list[npt.NDArray[np.floating]] = []
    scales: list[float] = []
    for path in tqdm.tqdm(paths, desc=desc):
        cloud, scale = normalize(np.load(path))
        clouds.append(cloud.astype(np.float32))
        scales.append(scale)

    return Clouds(clouds=np.stack(clouds, axis=0), scales=np.array(scales, dtype=np.float32))
