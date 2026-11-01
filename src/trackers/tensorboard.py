"""TensorBoard tracker handling the events of this project."""

import functools

from typing import override

import torch

from drytorch.core import log_events
from drytorch.trackers.tensorboard import TensorBoard

from src.lib.events import GenerationEvent, LatentSpaceEvent, ReconstructionEvent

__all__ = ['AutoencoderTensorBoard']


class AutoencoderTensorBoard(TensorBoard):
    """TensorBoard tracker that also writes 3D point clouds."""

    @functools.singledispatchmethod
    @override
    def notify(self, event: log_events.Event) -> None:
        return super().notify(event)

    @notify.register
    def _(self, event: LatentSpaceEvent) -> None:
        if self._writer is None:
            return super().notify(event)

        vertices = torch.cat([event.latent, event.pseudo_latent], dim=0).unsqueeze(0)
        colors = torch.cat(
            [
                torch.tensor([76, 76, 230], dtype=torch.uint8).repeat(len(event.latent), 1),
                torch.tensor([230, 76, 76], dtype=torch.uint8).repeat(len(event.pseudo_latent), 1),
            ],
            dim=0,
        ).unsqueeze(0)
        name = f'{event.model_name}/{event.source_name}-latent_space'
        self.writer.add_mesh(name, vertices=vertices, colors=colors, global_step=event.epoch)
        return super().notify(event)

    @notify.register
    def _(self, event: ReconstructionEvent) -> None:
        if self._writer is None:
            return super().notify(event)

        clouds = event.outputs.recon
        prefix = 'double_recon' if event.is_double else 'recon'
        for i, cloud in enumerate(clouds):
            name = f'{event.source_name}/{prefix}_{i}'
            self.writer.add_mesh(name, vertices=cloud.unsqueeze(0), global_step=event.epoch)

        return super().notify(event)

    @notify.register
    def _(self, event: GenerationEvent) -> None:
        if self._writer is None:
            return super().notify(event)

        clouds = event.outputs.recon
        for i, cloud in enumerate(clouds):
            name = f'{event.source_name}/gen_{i}'
            self.writer.add_mesh(name, vertices=cloud.unsqueeze(0), global_step=event.epoch)

        return super().notify(event)
