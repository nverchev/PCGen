"""Training of the codeword phase."""

from typing import TYPE_CHECKING, Any


from drytorch import Trainer

from src.config import Experiment, ExperimentConfig, hydra_main
from src.trackers import get_trackers
from src.config.environment import DEBUG_MODE
from src.data import WDataset, get_splits
from src.module import WAutoEncoder
from src.lib import (
    VQVAEModel,
    get_vqvae_model,
    get_w_encoder_learning_schema,
    get_w_encoder_loader,
    get_w_encoder_loss,
    get_w_encoder_model,
    load_trained_state,
    register_pruning,
    register_pseudo_latent_update,
)

if TYPE_CHECKING:
    from optuna import Trial
else:
    Trial = Any


def train_w_encoder(model: VQVAEModel, /, trial: Trial | None = None) -> None:
    """Train the codeword prior on the codes of a trained autoencoder.

    Args:
        model: trained autoencoder model, whose codes the prior learns.
        trial: Optuna trial, which replaces saving with pruning.

    Raises:
        TypeError: if the codeword model does not wrap a codeword prior.

    Side Effects:
        Resumes the prior from the checkpoint that ``user.load`` selects, if
        any. Saves the prior under its own name, leaving the autoencoder
        untouched, unless a trial is given.
    """
    cfg = Experiment.get_config()
    assert cfg.w_autoencoding is not None
    cfg_train = cfg.w_autoencoding.train
    train_split, eval_split = get_splits()
    w_model = get_w_encoder_model()
    w_autoencoder = w_model.module
    if not isinstance(w_autoencoder, WAutoEncoder):
        msg = 'The codeword model is not a codeword prior.'
        raise TypeError(msg)

    loss = get_w_encoder_loss(model.module)
    builder = Trainer.builder(
        w_model,
        learning_schema=get_w_encoder_learning_schema(),
        loader=get_w_encoder_loader(WDataset(train_split, model)),
        loss=loss,
    )
    register_pseudo_latent_update(builder, w_autoencoder)
    eval_loader = get_w_encoder_loader(WDataset(eval_split, model))
    if not cfg.final:
        builder.add_validation(eval_loader)

    if trial is not None:
        register_pruning(builder, trial, window=cfg_train.pruning_window)

    trainer = builder.build()
    if cfg.user.load:
        trainer.load_checkpoint(cfg.user.load)

    trainer.train_until(cfg_train.n_epochs)
    if trial is not None:
        return

    trainer.save_checkpoint()
    return


def setup_and_train(cfg: ExperimentConfig) -> None:
    """Set up the experiment and start the codeword training.

    Args:
        cfg: root configuration object.
    """
    trackers = get_trackers(cfg)
    exp = Experiment(cfg, name=cfg.name, par_dir=cfg.par_dir, tags=cfg.tags)
    for tracker in trackers:
        exp.trackers.subscribe(tracker)

    # the weights of the autoencoder live in its own run, which this one continues
    with exp.create_run(resume=True, run_id=cfg.user.run_id, record=not DEBUG_MODE):
        autoencoder = get_vqvae_model(mixed_precision=cfg.autoencoding.train.mixed_precision)
        load_trained_state(autoencoder)
        train_w_encoder(autoencoder)

    return


@hydra_main
def main(cfg: ExperimentConfig) -> None:
    """Entry point for the codeword training."""
    setup_and_train(cfg)
    return


if __name__ == '__main__':
    main()
