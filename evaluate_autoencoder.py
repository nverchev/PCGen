"""Reconstruction metrics of the trained autoencoder.

The evaluation partition is the test set only with ``final=true``, the validation set otherwise.

Usage::

    uv run evaluate_autoencoder.py
    uv run evaluate_autoencoder.py autoencoding.model.model_head=oracle  # the oracle needs no training
"""

from drytorch import Test

from src.config import Experiment, ExperimentConfig, hydra_main
from src.config.environment import DEBUG_MODE
from src.config.options import ModelHeads
from src.data import Partitions, get_split
from src.lib import get_autoencoder_loader, get_autoencoder_model, get_reconstruction_metrics, load_trained_state
from src.trackers import get_trackers


def evaluate_autoencoder() -> None:
    """Test the reconstructions of the trained autoencoder, or of the oracle.

    Side Effects:
        Reads the checkpoint that ``user.load`` selects, the last one by default.
    """
    cfg = Experiment.get_config()
    model = get_autoencoder_model()
    if cfg.autoencoding.model.model_head != ModelHeads.ORACLE:
        load_trained_state(model, epoch=cfg.user.load or -1)

    split = get_split(Partitions.test if cfg.final else Partitions.val)
    Test(model, loader=get_autoencoder_loader(split), metric=get_reconstruction_metrics())()
    return


@hydra_main
def main(cfg: ExperimentConfig) -> None:
    """Set up the experiment and evaluate the autoencoder."""
    trackers = get_trackers(cfg)
    exp = Experiment(cfg, name=cfg.name, par_dir=cfg.par_dir, tags=cfg.tags)
    for tracker in trackers:
        exp.trackers.subscribe(tracker)

    with exp.create_run(resume=True, run_id=cfg.user.run_id, record=not DEBUG_MODE):
        evaluate_autoencoder()

    return


if __name__ == '__main__':
    main()
