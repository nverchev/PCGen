"""Package for configuring the experiment."""

from src.config.environment import PROJECT_NAME, VERSION, ConfigPath
from src.config.experiment import Experiment
from src.config.hydra import DEFAULT_CONFIG, get_experiment_config, hydra_main
from src.config.specs import (
    AutoencodingConfig,
    ExperimentConfig,
    PhaseConfig,
    WAutoencodingConfig,
)
from src.config.torch import ActClass, NormClass

__all__ = [
    'DEFAULT_CONFIG',
    'PROJECT_NAME',
    'VERSION',
    'ActClass',
    'AutoencodingConfig',
    'ConfigPath',
    'Experiment',
    'ExperimentConfig',
    'NormClass',
    'PhaseConfig',
    'WAutoencodingConfig',
    'get_experiment_config',
    'hydra_main',
]
