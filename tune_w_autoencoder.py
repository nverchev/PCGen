"""Tuning of the codeword phase."""

from collections.abc import Callable

import hydra
import optuna
import torch

from drytorch.contrib.optuna import get_final_value, suggest_overrides
from drytorch.core.exceptions import ConvergenceError
from drytorch.core.registering import register_model, unregister_model
from omegaconf import DictConfig

from src.config import ConfigPath, Experiment, ExperimentConfig, get_experiment_config
from src.trackers import set_tuning_logging
from src.lib import VQVAEModel, get_vqvae_model, load_trained_state
from src.utils.tuning import create_study, impute_failed_trial, impute_pruned_trial
from train_w_autoencoder import train_w_encoder


def get_objective(tune_cfg: DictConfig, /) -> Callable[[optuna.Trial], float]:
    """Get the objective of the study: the validation loss of a short training of the prior.

    The trials encode the autoencoder trained with the configuration and the
    fixed overrides, which is loaded once and registered to each trial in turn.

    Args:
        tune_cfg: tuning configuration.

    Returns:
        A function training the codeword prior with the settings of a trial.
    """
    cfg = get_experiment_config(tune_cfg.config_name, overrides=list(tune_cfg.overrides))
    autoencoder = _load_autoencoder(cfg)
    fixed_overrides = [f'w_autoencoding.train.n_epochs={tune_cfg.n_epochs}']

    def objective(trial: optuna.Trial) -> float:
        overrides = [*fixed_overrides, *suggest_overrides(tune_cfg, trial)]
        cfg = get_experiment_config(tune_cfg.config_name, overrides=overrides)
        exp = Experiment(cfg, name=cfg.name, par_dir=cfg.par_dir, tags=cfg.tags)
        with exp.create_run(record=False):
            register_model(autoencoder)
            try:
                train_w_encoder(autoencoder, trial=trial)
            except optuna.TrialPruned:
                return impute_pruned_trial(trial)

            except ConvergenceError:
                return impute_failed_trial(trial)

            finally:
                unregister_model(autoencoder)
                if torch.accelerator.is_available():
                    torch.accelerator.empty_cache()

        return get_final_value(trial)

    return objective


def _load_autoencoder(cfg: ExperimentConfig) -> VQVAEModel:
    exp = Experiment(cfg, name=cfg.name, par_dir=cfg.par_dir, tags=cfg.tags)
    with exp.create_run(resume=True, run_id=cfg.user.run_id, record=False):
        # the codewords are encoded in the precision the autoencoder trained in
        autoencoder = get_vqvae_model(mixed_precision=cfg.autoencoding.train.mixed_precision)
        load_trained_state(autoencoder, epoch=cfg.user.load or -1)
        unregister_model(autoencoder)

    return autoencoder


@hydra.main(version_base=None, config_path=ConfigPath.TUNING_W_ENCODER.absolute(), config_name='defaults')
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
