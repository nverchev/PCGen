"""Reconstruction metrics of a trained autoencoder through its codeword prior."""

from drytorch import Test

from src.config import Experiment, ExperimentConfig, hydra_main
from src.trackers import get_trackers
from src.data import Partitions, get_split
from src.lib import (
    get_autoencoder_loader,
    get_reconstruction_metrics,
    get_vqvae_vae_model,
    load_vqvae_vae_module,
)


def evaluate_double_reconstruction() -> None:
    """Measure the double reconstructions of a finished run.

    The clouds are reconstructed from the codes nearest to the codeword the
    prior reconstructs, on the test split with ``final`` and on the
    validation split otherwise.

    Raises:
        ValueError: if the configured head has no prior.

    Side Effects:
        Reads the checkpoints of the run and records the evaluation.
    """
    cfg = Experiment.get_config()
    model = get_vqvae_vae_model(load_vqvae_vae_module())
    split = get_split(Partitions.test if cfg.final else Partitions.val)
    loader = get_autoencoder_loader(split)
    Test(model, loader=loader, metric=get_reconstruction_metrics())()
    return


@hydra_main
def main(cfg: ExperimentConfig) -> None:
    """Entry point for measuring double reconstructions."""
    trackers = get_trackers(cfg)
    exp = Experiment(cfg, name=cfg.name, par_dir=cfg.par_dir, tags=cfg.tags)
    for tracker in trackers:
        exp.trackers.subscribe(tracker)

    with exp.create_run(resume=True, run_id=cfg.user.run_id):
        evaluate_double_reconstruction()

    return


if __name__ == '__main__':
    main()
