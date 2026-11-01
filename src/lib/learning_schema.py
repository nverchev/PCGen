"""Learning schemas of the two training phases."""

from drytorch.lib import learning, schedulers

from src.config import Experiment
from src.config.options import Schedulers
from src.config.specs import LearningConfig, SchedulerConfig

__all__ = [
    'get_autoencoder_learning_schema',
    'get_scheduler',
    'get_w_encoder_learning_schema',
]


def get_autoencoder_learning_schema() -> learning.LearningSchema:
    """Get the learning schema of the autoencoder phase.

    Returns:
        The schema binding the optimizer, the learning rate and the schedule.
    """
    cfg_learn = Experiment.get_config().autoencoding.train.learn
    return _get_learning_schema(cfg_learn)


def get_scheduler(config: SchedulerConfig, /) -> schedulers.AbstractScheduler:
    """Get the learning rate schedule a phase asks for.

    Args:
        config: decay curve and its settings.

    Returns:
        The schedule of the configured curve.
    """
    if config.function == Schedulers.CONSTANT:
        return schedulers.ConstantScheduler()

    if config.function == Schedulers.COSINE:
        return schedulers.CosineScheduler(decay_steps=config.decay_steps, min_decay=config.min_decay)

    return schedulers.ExponentialScheduler(min_decay=config.min_decay)


def get_w_encoder_learning_schema() -> learning.LearningSchema:
    """Get the learning schema of the codeword phase.

    Returns:
        The schema binding the optimizer, the learning rate and the schedule.
    """
    cfg_w = Experiment.get_config().w_autoencoding
    assert cfg_w is not None
    return _get_learning_schema(cfg_w.train.learn)


def _get_learning_schema(config: LearningConfig, /) -> learning.LearningSchema:
    return learning.LearningSchema(
        optimizer_cls=config.optimizer,
        base_lr=config.lr,
        scheduler=get_scheduler(config.scheduler),
        optimizer_defaults={'weight_decay': config.weight_decay},
    )
