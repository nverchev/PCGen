"""Tuning of the autoencoder phase."""

from collections.abc import Callable

import hydra
import optuna
import torch

from drytorch.contrib.optuna import get_final_value, suggest_overrides
from drytorch.core.exceptions import ConvergenceError
from omegaconf import DictConfig

from src.config import ConfigPath, Experiment, get_experiment_config
from src.trackers import set_tuning_logging
from src.utils.tuning import create_study, impute_failed_trial, impute_pruned_trial
from train_autoencoder import train_autoencoder


def get_objective(tune_cfg: DictConfig, /) -> Callable[[optuna.Trial], float]:
    """Get the objective of the study: the validation loss of a short training.

    Args:
        tune_cfg: tuning configuration.

    Returns:
        A function training the autoencoder with the settings of a trial.
    """
    fixed_overrides = [f'autoencoding.train.n_epochs={tune_cfg.n_epochs}']

    def objective(trial: optuna.Trial) -> float:
        overrides = [*fixed_overrides, *suggest_overrides(tune_cfg, trial)]
        cfg = get_experiment_config(tune_cfg.config_name, overrides=overrides)
        exp = Experiment(cfg, name=cfg.name, par_dir=cfg.par_dir, tags=cfg.tags)
        with exp.create_run(record=False):
            try:
                train_autoencoder(trial=trial)
            except optuna.TrialPruned:
                return impute_pruned_trial(trial)

            except ConvergenceError:
                return impute_failed_trial(trial)

            finally:
                if torch.accelerator.is_available():
                    torch.accelerator.empty_cache()

        return get_final_value(trial)

    return objective


@hydra.main(version_base=None, config_path=ConfigPath.TUNING_AUTOENCODER.absolute(), config_name='defaults')
def main(tune_cfg: DictConfig) -> None:
    """Set up the study and launch the optimization."""
    set_tuning_logging()
    study = create_study(tune_cfg)
    # the trials of the study, from every worker and every run, stop at n_trials
    max_trials = optuna.study.MaxTrialsCallback(tune_cfg.tune.n_trials, states=None)
    study.optimize(get_objective(tune_cfg), n_trials=tune_cfg.tune.n_trials, callbacks=[max_trials])
    return


if __name__ == '__main__':
    main()
