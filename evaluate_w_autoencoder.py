"""Loss and latent space of the trained codeword prior.

The evaluation partition is the test set only with ``final=true``, the validation set otherwise.

Usage::

    uv run evaluate_w_autoencoder.py
"""

import torch

from drytorch import Test

from src.config import Experiment, ExperimentConfig, hydra_main
from src.config.environment import DEBUG_MODE
from src.data import Partitions, WDataset, get_split
from src.lib import get_vqvae_model, get_w_encoder_loader, get_w_encoder_loss, load_trained_state, load_w_encoder_model
from src.lib.events import LatentSpaceEvent
from src.module import WAutoEncoder
from src.trackers import get_trackers
from src.utils import project_to_3d


def evaluate_w_autoencoder() -> None:
    """Test the loss of the codeword prior and publish its latent space.

    Raises:
        TypeError: if the codeword model does not wrap a codeword prior.

    Side Effects:
        Reads the last checkpoints of the autoencoder and of the prior.
    """
    cfg = Experiment.get_config()
    autoencoder = get_vqvae_model(mixed_precision=cfg.autoencoding.train.mixed_precision)
    load_trained_state(autoencoder)
    w_model = load_w_encoder_model()
    w_autoencoder = w_model.module
    if not isinstance(w_autoencoder, WAutoEncoder):
        msg = 'The codeword model is not a codeword prior.'
        raise TypeError(msg)

    split = get_split(Partitions.test if cfg.final else Partitions.val)
    loader = get_w_encoder_loader(WDataset(split, autoencoder))
    test = Test(w_model, loader=loader, metric=get_w_encoder_loss(autoencoder.module))
    test(store_outputs=True)
    latent = torch.vstack([outputs.mu for outputs in test.outputs_list])
    latent_3d, pseudo_3d = project_to_3d(latent, w_autoencoder.pseudo_mu)
    LatentSpaceEvent(
        source_name=test.name,
        model_name=w_model.name,
        epoch=w_model.epoch,
        latent=latent_3d,
        pseudo_latent=pseudo_3d,
    )
    return


@hydra_main
def main(cfg: ExperimentConfig) -> None:
    """Set up the experiment and evaluate the codeword prior."""
    trackers = get_trackers(cfg)
    exp = Experiment(cfg, name=cfg.name, par_dir=cfg.par_dir, tags=cfg.tags)
    for tracker in trackers:
        exp.trackers.subscribe(tracker)

    with exp.create_run(resume=True, run_id=cfg.user.run_id, record=not DEBUG_MODE):
        evaluate_w_autoencoder()

    return


if __name__ == '__main__':
    main()
