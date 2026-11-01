"""Plotly tracker handling the events of this project."""

import functools

from typing import override

import plotly.graph_objs as go
import torch

from drytorch.core import log_events
from drytorch.trackers.plotly import PlotlyPlotter

from src.lib.events import LatentSpaceEvent

__all__ = ['AutoencoderPlotly']


class AutoencoderPlotly(PlotlyPlotter):
    """Plotly tracker that also plots the latent space in 3D."""

    @functools.singledispatchmethod
    @override
    def notify(self, event: log_events.Event) -> None:
        return super().notify(event)

    @notify.register
    def _(self, event: LatentSpaceEvent) -> None:
        fig = self.render_latent(
            event.latent,
            event.pseudo_latent,
            name=f'{event.model_name}/{event.source_name}-latent_space',
        )
        if self.record:
            folder = self._get_run_dir() / event.model_name
            folder.mkdir(parents=True, exist_ok=True)
            fig.write_html(folder / f'latent_space_epoch_{event.epoch}.html')

        if self.show:
            fig.show()

        return super().notify(event)

    @staticmethod
    def render_latent(
        latent: torch.Tensor,
        pseudo_latent: torch.Tensor,
        *,
        name: str = 'latent_space',
    ) -> go.Figure:
        """Render continuous latent space points and pseudo-inputs via Plotly 3D scatter.

        Args:
            latent: Continuous latent points tensor projected to 3D (N, 3).
            pseudo_latent: Pseudo-input latent points tensor projected to 3D (M, 3).
            name: Figure title.

        Returns:
            Plotly Figure object.
        """
        mu_np = latent.numpy()
        pseudo_np = pseudo_latent.numpy()
        fig = go.Figure(
            data=[
                go.Scatter3d(
                    x=mu_np[:, 0],
                    y=mu_np[:, 1],
                    z=mu_np[:, 2],
                    mode='markers',
                    marker={'size': 3, 'color': 'blue', 'opacity': 0.7},
                    name='Latent',
                ),
                go.Scatter3d(
                    x=pseudo_np[:, 0],
                    y=pseudo_np[:, 1],
                    z=pseudo_np[:, 2],
                    mode='markers',
                    marker={'size': 6, 'color': 'red', 'symbol': 'diamond'},
                    name='Pseudo-latent',
                ),
            ],
            layout=go.Layout(
                title=name,
                scene={'xaxis': {'title': 'Dim 0'}, 'yaxis': {'title': 'Dim 1'}, 'zaxis': {'title': 'Dim 2'}},
            ),
        )
        return fig
