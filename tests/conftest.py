"""Configuration module with shared test fixtures for PCGen."""

from collections.abc import Generator
import pathlib

import pytest

from src.config import Experiment, ExperimentConfig, get_experiment_config


@pytest.fixture(scope='package')
def default_config() -> ExperimentConfig:
    """Return the configuration without overrides."""
    return get_experiment_config()


@pytest.fixture
def overrides() -> list[str]:
    """Return the overrides of the configuration; a test parametrizes it to change settings."""
    return []


@pytest.fixture(autouse=True)
def active_experiment(
    overrides: list[str], default_config: ExperimentConfig, tmp_path: pathlib.Path
) -> Generator[Experiment]:
    """Provide an active run of the configuration with the overrides, for every test."""
    config = get_experiment_config(overrides=overrides) if overrides else default_config
    exp = Experiment(config, name='test_exp', par_dir=tmp_path)
    with exp.create_run(record=False):
        yield exp

    return
