"""Visualisation of double reconstructions through a trained codeword prior."""

from drytorch.lib import loading

from src.config import Experiment, ExperimentConfig, hydra_main
from src.config.options import Overlays
from src.trackers import get_trackers
from src.data import Partitions, get_split
from src.lib import (
    get_vqvae_vae_model,
    load_vqvae_vae_module,
)
from src.utils.visualization import reconstruct_and_publish


def visualize_double_reconstructions() -> None:
    """Reconstruct point clouds through the codeword prior and publish event.

    Raises:
        IndexError: if an index is outside the split.
        ValueError: if the configured head is not the quantized one.
    """
    cfg = Experiment.get_config()
    double_module = load_vqvae_vae_module()
    double_model = get_vqvae_vae_model(double_module)
    split = get_split(Partitions.test if cfg.final else Partitions.val)
    if cfg.user.plot.sample_indices:
        indices, (inputs, _) = loading.take_from_dataset(
            split, indices=cfg.user.plot.sample_indices, device=double_model.device
        )
    else:
        indices, (inputs, _) = loading.take_from_dataset(
            split, n_samples=cfg.autoencoding.train.batch_size, preserve_order=False, device=double_model.device
        )

    clouds = inputs.cloud
    reconstruct_and_publish(
        double_module,
        clouds,
        sample_indices=indices,
        epoch=double_model.epoch,
        loop=cfg.user.plot.overlay == Overlays.SAMPLING_LOOP,
        is_double=True,
        source_name='double_reconstruction',
    )
    return


@hydra_main
def main(cfg: ExperimentConfig) -> None:
    """Entry point for visualising double reconstructions."""
    trackers = get_trackers(cfg)
    exp = Experiment(cfg, name=cfg.name, par_dir=cfg.par_dir, tags=cfg.tags)
    for tracker in trackers:
        exp.trackers.subscribe(tracker)

    with exp.create_run(resume=True, run_id=cfg.user.run_id):
        visualize_double_reconstructions()

    return


if __name__ == '__main__':
    main()
