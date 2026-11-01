"""Visualisation of clouds drawn from the prior of a trained model."""

import torch

from src.config import Experiment, ExperimentConfig, hydra_main
from src.config.options import Overlays
from src.trackers import get_trackers
from src.module import VQVAE
from src.lib import get_autoencoder_model, load_trained_state, load_vqvae_vae
from src.utils.visualization import generate_and_publish


def generate_samples() -> None:
    """Draw clouds from the prior and publish GenerationEvent.

    Raises:
        ValueError: if the configured head has no prior to draw from.
    """
    cfg = Experiment.get_config()
    cfg_generation = cfg.user.generation
    model = get_autoencoder_model()
    load_trained_state(model, epoch=cfg.user.load or -1)
    module = model.module
    if not isinstance(module, VQVAE):
        msg = 'Only the quantized head draws clouds from a prior.'
        raise ValueError(msg)

    cfg_w = cfg.w_autoencoding
    assert cfg_w is not None
    z_bias = torch.zeros(cfg_generation.n_generated, cfg_w.model.z_dim, device=model.device)
    z_bias[:, cfg_generation.bias_dim] = cfg_generation.bias_value
    generate_and_publish(
        load_vqvae_vae(module),
        n_clouds=cfg_generation.n_generated,
        z_bias=z_bias,
        epoch=model.epoch,
        loop=cfg.user.plot.overlay == Overlays.SAMPLING_LOOP,
    )
    return


@hydra_main
def main(cfg: ExperimentConfig) -> None:
    """Entry point for drawing and visualising clouds."""
    trackers = get_trackers(cfg)
    exp = Experiment(cfg, name=cfg.name, par_dir=cfg.par_dir, tags=cfg.tags)
    for tracker in trackers:
        exp.trackers.subscribe(tracker)

    with exp.create_run(resume=True, run_id=cfg.user.run_id):
        generate_samples()

    return


if __name__ == '__main__':
    main()
