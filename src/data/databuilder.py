"""Base builder preparing a point cloud dataset for its splits."""

import abc
import json
import pathlib

from typing import ClassVar, Final

from src.config import Experiment
from src.data.split import CloudSplit, Partitions

__all__ = ['DataBuilder']


class DataBuilder(abc.ABC):
    """Dataset preparing its clouds once and handing them to the splits.

    A subclass states where the clouds come from and how they are stored, and
    gives each partition's clouds to a split that samples them.

    Attributes:
        classes: Synset code of every class the splits draw from.
        dir_processed: Directory holding the prepared clouds.
        dir_raw: Directory holding the clouds as they were distributed.
    """

    folder_processed: ClassVar[str]
    folder_raw: ClassVar[str]
    metadata_file: ClassVar[str]

    classes: Final[dict[str, str]]
    dir_processed: Final[pathlib.Path]
    dir_raw: Final[pathlib.Path]

    def __init__(self) -> None:
        """Initialize.

        Raises:
            ValueError: if the configuration selects a class the dataset misses.
        """
        cfg = Experiment.get_config()
        cfg_path = cfg.user.path
        self.dir_processed = cfg_path.data_dir / self.folder_processed
        self.dir_raw = cfg_path.data_dir / self.folder_raw
        with open(cfg_path.metadata_dir / self.metadata_file) as file:
            metadata: dict[str, str] = json.load(file)

        selected = [str(name) for name in cfg.data.dataset.select_classes]
        unknown = [name for name in selected if name not in metadata.values()]
        if unknown:
            msg = f'{unknown} are not classes of {type(self).__name__}.'
            raise ValueError(msg)

        codes = {name: code for code, name in metadata.items()}
        self.classes = {name: codes[name] for name in selected or sorted(codes)}
        self._prepare()
        return

    @abc.abstractmethod
    def get_split(self, partition: Partitions) -> CloudSplit:
        """Get the split of a partition.

        Args:
            partition: partition to draw the clouds from.

        Returns:
            A split over the clouds of that partition.
        """

    @abc.abstractmethod
    def _prepare(self) -> None:
        """Make the clouds available and index them for the splits."""
