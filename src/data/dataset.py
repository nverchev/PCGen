"""Selection of the active dataset and construction of its splits."""

from src.config import Experiment
from src.config.options import Datasets
from src.data.coins import CoinsSplit
from src.data.shapenet_atlas import ShapeNetAtlasDataBuilder
from src.data.shapenet_flow import ShapeNetFlowDataBuilder
from src.data.split import DataSplit, Partitions

__all__ = [
    'get_databuilder',
    'get_split',
    'get_splits',
]


def get_databuilder() -> ShapeNetAtlasDataBuilder | ShapeNetFlowDataBuilder:
    """Get the builder of the dataset the configuration selects, prepared and indexed.

    Returns:
        A builder handing out the splits of the selected dataset.

    Raises:
        ValueError: if the configuration selects the synthetic dataset, which
            has nothing to prepare and is a split of its own.
    """
    name = Experiment.get_config().data.dataset.name
    if name == Datasets.SHAPENET_FLOW:
        return ShapeNetFlowDataBuilder()

    if name == Datasets.SHAPENET_ATLAS:
        return ShapeNetAtlasDataBuilder()

    msg = f'{name} needs no builder: build its split directly.'
    raise ValueError(msg)


def get_split(partition: Partitions) -> DataSplit:
    """Get one split, preparing the dataset it comes from.

    Args:
        partition: partition to draw the clouds from.

    Returns:
        A split over the clouds of that partition.
    """
    name = Experiment.get_config().data.dataset.name
    if name == Datasets.COINS:
        return CoinsSplit()

    return get_databuilder().get_split(partition)


def get_splits() -> tuple[DataSplit, DataSplit]:
    """Get the training and the evaluation split, one process at a time.

    Preparing a dataset writes files, so the processes take their turn to build
    the splits and the others wait for the files to be there.

    Returns:
        The split to train on and the split to evaluate on.

    Raises:
        RuntimeError: if no process built the splits.
    """
    cfg = Experiment.get_config()
    splits: tuple[DataSplit, DataSplit] | None = None
    if cfg.user.n_subprocesses:
        from torch import distributed as dist

        rank = dist.get_rank()
        for turn in range(cfg.user.n_subprocesses):
            if rank == turn:
                splits = _get_splits()

            dist.barrier()
    else:
        splits = _get_splits()

    if splits is None:
        raise RuntimeError('Splits could not be created.')

    return splits


def _get_splits() -> tuple[DataSplit, DataSplit]:
    cfg = Experiment.get_config()
    train_partition = Partitions.train_val if cfg.final else Partitions.train
    eval_partition = Partitions.test if cfg.final else Partitions.val
    if cfg.data.dataset.name == Datasets.COINS:
        return CoinsSplit(), CoinsSplit()

    databuilder = get_databuilder()
    return databuilder.get_split(train_partition), databuilder.get_split(eval_partition)
