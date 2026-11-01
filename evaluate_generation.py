"""Scoring of the clouds a trained model generates against a reference set."""

import logging

from typing import Any

import torch

from drytorch.lib import models
from torch import nn

from src.config import Experiment, ExperimentConfig, hydra_main
from src.trackers import get_trackers
from src.config.options import ModelHeads
from src.data import Partitions, get_split
from src.lib import (
    get_generation_model,
    get_vqvae_vae_model,
    load_vqvae_vae_module,
)
from src.utils.distances import chamfer, emd
from src.module.generation import Generation, Resampling
from src.utils.generation import GenerativeScores

logger = logging.getLogger(__name__)


@torch.inference_mode()
def evaluate_generation() -> None:
    """Score generated clouds against the reference clouds, over several draws.

    The scores vary with the draw, so each distance is scored on as many draws
    as its number of repeats asks for and the spread is reported.

    Raises:
        ValueError: if the configured head neither generates nor resamples.

    Side Effects:
        Every draw publishes its scores as metrics of the generating model.
    """
    cfg = Experiment.get_config()
    cfg_evaluation = cfg.evaluation
    reference = _get_clouds(Partitions.test if cfg.final else Partitions.val)
    model: models.Model[Any, Any]
    if cfg.autoencoding.model.model_head == ModelHeads.ORACLE:
        train_clouds = _get_clouds(Partitions.train_val if cfg.final else Partitions.train)
        model = get_generation_model(Resampling(torch.stack(train_clouds)), name='oracle')
        drawing: nn.Module = model.module
    else:
        vqvae_vae = load_vqvae_vae_module()
        model = get_vqvae_vae_model(vqvae_vae)
        drawing = Generation(vqvae_vae)

    batch_size = cfg.autoencoding.train.batch_size
    for name, distance, n_repeats in (
        ('Chamfer', chamfer, cfg_evaluation.n_chamfer_repeats),
        ('EMD', emd, cfg_evaluation.n_emd_repeats),
    ):
        if not n_repeats:
            logger.info(f'No test performed for {name}.')
            continue

        scores = GenerativeScores(reference, distance, name=f'Generation_{name}', device=model.device)
        scored: list[dict[str, float]] = []
        for _ in range(n_repeats):
            generated = _draw_clouds(drawing, len(reference), batch_size=batch_size, device=model.device)
            scored.append(scores(model, generated))

        _report(name, scored)

    return


@hydra_main
def main(cfg: ExperimentConfig) -> None:
    """Entry point for scoring generated clouds."""
    trackers = get_trackers(cfg)
    exp = Experiment(cfg, name=cfg.name, par_dir=cfg.par_dir, tags=cfg.tags)
    for tracker in trackers:
        exp.trackers.subscribe(tracker)

    with exp.create_run(resume=True, run_id=cfg.user.run_id):
        evaluate_generation()

    return


def _draw_clouds(drawing: nn.Module, n_clouds: int, *, batch_size: int, device: torch.device) -> list[torch.Tensor]:
    clouds: list[torch.Tensor] = []
    for start in range(0, n_clouds, batch_size):
        indices = torch.arange(start, min(start + batch_size, n_clouds), device=device)
        clouds.extend(drawing(indices).cpu())

    return clouds


def _get_clouds(partition: Partitions) -> list[torch.Tensor]:
    split = get_split(partition)
    return [split[index][0].cloud for index in range(len(split))]


def _report(distance: str, scored: list[dict[str, float]]) -> None:
    for score in scored[0]:
        values = torch.tensor([one[score] for one in scored])
        spread = values.std().item() if values.numel() > 1 else 0.0
        logger.info(
            f'{score} ({distance}) over {values.numel()} tests: '
            f'min {values.min().item():.4e} max {values.max().item():.4e} '
            f'mean {values.mean().item():.4e} std {spread:.4e}'
        )

    return


if __name__ == '__main__':
    main()
