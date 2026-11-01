"""Weights & Biases tracker handling the events of this project."""

import functools

from typing import override

import numpy as np
import wandb

from drytorch.core import log_events
from drytorch.trackers.wandb import Wandb

from src.lib.events import GenerationEvent, LatentSpaceEvent, ReconstructionEvent

__all__ = ['AutoencoderWandb']


class AutoencoderWandb(Wandb):
    """Weights & Biases tracker that also logs 3D point clouds."""

    @functools.singledispatchmethod
    @override
    def notify(self, event: log_events.Event) -> None:
        return super().notify(event)

    @notify.register
    def _(self, event: LatentSpaceEvent) -> None:
        if self._run is None:
            return super().notify(event)

        latent_pts = np.hstack([event.latent.numpy(), np.tile([76, 76, 230], (len(event.latent), 1))])
        pseudo_pts = np.hstack([event.pseudo_latent.numpy(), np.tile([230, 76, 76], (len(event.pseudo_latent), 1))])
        pts = np.vstack([latent_pts, pseudo_pts])
        name = f'{event.model_name}/{event.source_name}-latent_space'
        self.log_at_epoch({name: wandb.Object3D(pts)}, model_name=event.model_name, epoch=event.epoch)
        return super().notify(event)

    @notify.register
    def _(self, event: ReconstructionEvent) -> None:
        if self._run is None:
            return super().notify(event)

        clouds = event.outputs.recon.detach().cpu().numpy()
        prefix = 'double_recon' if event.is_double else 'recon'
        clouds_3d = {f'{event.source_name}/{prefix}_{i}': wandb.Object3D(cloud) for i, cloud in enumerate(clouds)}
        self.log_at_epoch(clouds_3d, model_name=event.model_name, epoch=event.epoch)
        return super().notify(event)

    @notify.register
    def _(self, event: GenerationEvent) -> None:
        if self._run is None:
            return super().notify(event)

        clouds = event.outputs.recon.detach().cpu().numpy()
        clouds_3d = {f'{event.source_name}/gen_{i}': wandb.Object3D(cloud) for i, cloud in enumerate(clouds)}
        self.log_at_epoch(clouds_3d, model_name=event.model_name, epoch=event.epoch)
        return super().notify(event)
