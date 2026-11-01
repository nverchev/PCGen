"""Training of the autoencoder phase."""

from typing import TYPE_CHECKING, Any

from drytorch import Trainer
from drytorch.lib import evaluations

from src.config import Experiment, ExperimentConfig, hydra_main
from src.trackers import get_trackers
from src.config.environment import DEBUG_MODE
from src.config.options import ModelHeads, Overlays
from src.data import get_splits
from src.module import VQVAE
from src.lib import (
    get_autoencoder_learning_schema,
    get_autoencoder_loader,
    get_autoencoder_loss,
    get_autoencoder_model,
    register_checkpointing,
    register_codebook_revival,
    register_pruning,
    register_reconstruction_logging,
)

if TYPE_CHECKING:
    from optuna import Trial
else:
    Trial = Any


def train_autoencoder(trial: Trial | None = None) -> None:
    """Train the autoencoder and save it.

    Args:
        trial: Optuna trial, which replaces checkpointing with pruning.

    Raises:
        ValueError: if the configured head is the oracle, which has nothing to train.
    """
    cfg = Experiment.get_config()
    cfg_train = cfg.autoencoding.train
    if cfg.autoencoding.model.model_head == ModelHeads.ORACLE:
        msg = 'The oracle head has nothing to train: evaluate it with evaluate_autoencoder.py.'
        raise ValueError(msg)

    train_split, eval_split = get_splits()
    model = get_autoencoder_model(mixed_precision=cfg_train.mixed_precision)

    loss = get_autoencoder_loss()
    train_loader = get_autoencoder_loader(train_split)
    builder = Trainer.builder(
        model,
        learning_schema=get_autoencoder_learning_schema(),
        loader=train_loader,
        loss=loss,
    )
    if not cfg.final:
        builder.add_validation(get_autoencoder_loader(eval_split))

    if isinstance(model.module, VQVAE):
        diagnostic = evaluations.Evaluation(model, loader=train_loader, metric=loss.copy(), name='Diagnostic')
        register_codebook_revival(
            builder,
            diagnostic=diagnostic,
            module=model.module,
            noise=cfg.autoencoding.model.vq_noise,
            revive_every=cfg_train.checkpoint_every,
        )

    if trial is None:
        register_checkpointing(builder, checkpoint_every=cfg_train.checkpoint_every)
        cfg_viz = cfg.user.plot
        if cfg_viz.reconstruction_every:
            register_reconstruction_logging(
                builder,
                n_samples=cfg_train.batch_size,
                split=eval_split,
                sample_indices=cfg_viz.sample_indices,
                reconstruction_every=cfg_viz.reconstruction_every,
                loop=cfg_viz.overlay == Overlays.SAMPLING_LOOP,
            )
    else:
        register_pruning(builder, trial, window=cfg_train.pruning_window)

    trainer = builder.build()
    if cfg.user.load:
        trainer.load_checkpoint(cfg.user.load)

    trainer.train_until(cfg_train.n_epochs)
    if trial is None:
        trainer.save_checkpoint()

    return


def setup_and_train(cfg: ExperimentConfig) -> None:
    """Set up the experiment and start the autoencoder training.

    Args:
        cfg: root configuration object.
    """
    trackers = get_trackers(cfg)
    exp = Experiment(cfg, name=cfg.name, par_dir=cfg.par_dir, tags=cfg.tags)
    for tracker in trackers:
        exp.trackers.subscribe(tracker)

    resume = cfg.user.load != 0 or cfg.user.run_id is not None
    with exp.create_run(resume=resume, run_id=cfg.user.run_id, record=not DEBUG_MODE):
        train_autoencoder()

    return


@hydra_main
def main(cfg: ExperimentConfig) -> None:
    """Entry point for the autoencoder training."""
    n_processes = cfg.user.n_subprocesses
    if n_processes:
        from src.utils import DistributedWorker

        DistributedWorker(setup_and_train, n_processes).spawn(cfg)
    else:
        setup_and_train(cfg)

    return


if __name__ == '__main__':
    main()
