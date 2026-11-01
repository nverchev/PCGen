"""Setup of the trackers of the experiment."""

from __future__ import annotations

import os
import warnings

from typing import TYPE_CHECKING

from hydra.core import utils as hydra_utils

import drytorch

from drytorch.core import tracking
from drytorch.trackers import logging

from src.config.environment import DEBUG_MODE
from src.config.specs import ExperimentConfig


if TYPE_CHECKING:
    from drytorch.trackers.sqlalchemy import SQLConnection

__all__ = [
    'get_metric_database',
    'get_trackers',
    'set_tuning_logging',
]


def get_metric_database(cfg: ExperimentConfig) -> SQLConnection | None:
    """Connect to the metric database of the configuration.

    Args:
        cfg: Root configuration object.

    Returns:
        The connection, or ``None`` if the tracker is off or SQLAlchemy is not installed.

    Side Effects:
        Creates the database in ``par_dir`` if it does not exist.
    """
    if not cfg.user.trackers.sqlalchemy:
        return None

    try:
        import sqlalchemy

        from drytorch.trackers.sqlalchemy import SQLConnection

    except ImportError as err:
        warnings.warn(f'SQLAlchemy not installed, skipping. {err}', stacklevel=2)
        return None

    engine_path = cfg.par_dir / 'metrics.db'
    cfg.par_dir.mkdir(parents=True, exist_ok=True)
    engine = sqlalchemy.create_engine(f'sqlite:///{engine_path}')
    return SQLConnection(engine=engine)


def get_trackers(cfg: ExperimentConfig, debug_mode: bool = DEBUG_MODE) -> list[tracking.Tracker]:
    """Instantiate the experiment trackers requested in *cfg*.

    Trackers are only activated outside debug mode.  In debug mode an empty
    list is returned so that no files or remote metrics are written.

    Args:
        cfg: Root configuration object.
        debug_mode: When ``True``, skip all tracker initialisation. Defaults to
            whether a debugger runs the process.

    Returns:
        A list of initialised :class:`drytorch.core.tracking.Tracker` objects
        ready to be passed to :meth:`drytorch.Experiment.trackers.subscribe`.
    """
    cfg_trackers = cfg.user.trackers
    cfg_hydra = cfg.user.hydra
    hydra_utils.configure_log(cfg_hydra.job_logging)
    drytorch.init_trackers(mode='hydra')
    tracker_list: list[tracking.Tracker] = []
    if debug_mode:
        return tracker_list

    if cfg_trackers.wandb:
        try:
            import wandb

            from src.trackers.wandb import AutoencoderWandb

        except ImportError as err:
            warnings.warn(f'Wandb not installed, skipping. {err}', stacklevel=2)

        else:
            tracker_list.append(AutoencoderWandb(settings=wandb.Settings(project=cfg.project)))

    if cfg_trackers.hydra:
        from drytorch.trackers.hydra import HydraLink

        tracker_list.append(HydraLink(hydra_dir=cfg.user.hydra.output_dir))

    if cfg_trackers.csv:
        from drytorch.trackers.csv import CSVDumper

        tracker_list.append(CSVDumper())

    if cfg_trackers.tensorboard:
        try:
            from src.trackers.tensorboard import AutoencoderTensorBoard

        except ImportError as err:
            warnings.warn(f'TensorBoard not installed, skipping. {err}', stacklevel=2)

        else:
            tracker_list.append(AutoencoderTensorBoard())

    sql_connection = get_metric_database(cfg)
    if sql_connection is not None:
        tracker_list.append(sql_connection)

    if cfg_trackers.pyvista:
        try:
            from src.trackers.pyvista import PyVistaTracker

        except ImportError as err:
            warnings.warn(f'PyVista not installed, skipping. {err}', stacklevel=2)

        else:
            tracker_list.append(
                PyVistaTracker(
                    show=cfg.user.plot.show,
                    overlay=cfg.user.plot.overlay,
                    sample_indices=cfg.user.plot.sample_indices,
                    class_names=[str(name) for name in cfg.data.dataset.select_classes],
                    m_test=cfg.autoencoding.train.m_test,
                    n_components=cfg.autoencoding.architecture.decoder.n_components,
                )
            )

    if cfg_trackers.plotly:
        try:
            from src.trackers.plotly import AutoencoderPlotly

        except ImportError as err:
            warnings.warn(f'Plotly not installed, skipping. {err}', stacklevel=2)

        else:
            tracker_list.append(AutoencoderPlotly(show=cfg.user.plot.show))

    return tracker_list


def set_tuning_logging() -> None:
    """Configure minimal console logging for parallel Optuna workers.

    When GNU parallel runs the workers, it sets ``PARALLEL_SEQ`` in each of them, and the
    output is restricted to test metrics to prevent interleaved logs across workers.
    """
    if 'PARALLEL_SEQ' in os.environ:
        drytorch.remove_all_default_trackers()
        drytorch.extend_default_trackers([logging.BuiltinLogger()])
        logging.set_verbosity(logging.INFO_LEVELS.test)
    else:
        drytorch.init_trackers(mode='minimal')

    return
