"""Visualisation of the reconstructions of a trained model."""

from drytorch.lib import loading

from src.config import Experiment, ExperimentConfig, hydra_main
from src.config.options import Overlays
from src.trackers import get_trackers
from src.data import Partitions, get_split
from src.module import BaseModel
from src.lib import get_autoencoder_model, load_trained_state
from src.utils.visualization import reconstruct_and_publish


def visualize_reconstructions() -> None:
    """Reconstruct point clouds for configured sample indices and publish event.

    Raises:
        IndexError: if an index is outside the split.
        ValueError: if the loaded module is not a point cloud autoencoder.
    """
    cfg = Experiment.get_config()
    model = get_autoencoder_model()
    load_trained_state(model, epoch=cfg.user.load or -1)
    module = model.module
    if not isinstance(module, BaseModel):
        msg = 'The loaded module does not reconstruct point clouds.'
        raise ValueError(msg)

    split = get_split(Partitions.test if cfg.final else Partitions.val)
    if cfg.user.plot.sample_indices:
        indices, (inputs, _) = loading.take_from_dataset(
            split, indices=cfg.user.plot.sample_indices, device=model.device
        )
    else:
        indices, (inputs, _) = loading.take_from_dataset(
            split, n_samples=cfg.autoencoding.train.batch_size, preserve_order=False, device=model.device
        )

    clouds = inputs.cloud
    reconstruct_and_publish(
        module,
        clouds,
        sample_indices=indices,
        epoch=model.epoch,
        loop=cfg.user.plot.overlay == Overlays.SAMPLING_LOOP,
    )
    return


@hydra_main
def main(cfg: ExperimentConfig) -> None:
    """Entry point for visualising reconstructions."""
    trackers = get_trackers(cfg)
    exp = Experiment(cfg, name=cfg.name, par_dir=cfg.par_dir, tags=cfg.tags)
    for tracker in trackers:
        exp.trackers.subscribe(tracker)

    with exp.create_run(resume=True, run_id=cfg.user.run_id):
        visualize_reconstructions()

    return


if __name__ == '__main__':
    main()
