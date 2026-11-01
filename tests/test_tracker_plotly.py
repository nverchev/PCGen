"""Tests for Plotly tracker and rendering."""

import pathlib

import plotly.graph_objs as go
from pytest_mock import MockerFixture
import torch

from src.config import Experiment
from src.lib.events import LatentSpaceEvent
from src.trackers.plotly import AutoencoderPlotly


def test_render_latent_produces_figure() -> None:
    """render_latent produces a plotly Figure."""
    latent = torch.randn(8, 3)
    pseudo = torch.randn(4, 3)

    fig = AutoencoderPlotly.render_latent(latent, pseudo)

    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 2


def test_latent_space_event_notifies_plotly(
    active_experiment: Experiment, tmp_path: pathlib.Path, mocker: MockerFixture
) -> None:
    """Publishing LatentSpaceEvent notifies AutoencoderPlotly."""
    plotly_tracker = AutoencoderPlotly()
    mocker.patch.object(plotly_tracker, '_get_run_dir', return_value=tmp_path)
    active_experiment.trackers.subscribe(plotly_tracker)
    mock_render = mocker.patch.object(AutoencoderPlotly, 'render_latent')
    latent = torch.zeros(2, 3)
    pseudo = torch.ones(2, 3)

    LatentSpaceEvent(
        source_name='Test',
        model_name='VQVAE',
        epoch=1,
        latent=latent,
        pseudo_latent=pseudo,
    )

    mock_render.assert_called_once_with(
        latent,
        pseudo,
        name='VQVAE/Test-latent_space',
    )


def test_latent_space_event_saves_html(
    active_experiment: Experiment, tmp_path: pathlib.Path, mocker: MockerFixture
) -> None:
    """Publishing LatentSpaceEvent writes an HTML file under the tracker run dir and respects show."""
    plotly_tracker = AutoencoderPlotly()
    mocker.patch.object(plotly_tracker, '_get_run_dir', return_value=tmp_path)
    active_experiment.trackers.subscribe(plotly_tracker)
    latent = torch.zeros(2, 3)
    pseudo = torch.ones(2, 3)

    mock_show = mocker.patch.object(go.Figure, 'show')

    LatentSpaceEvent(
        source_name='Test',
        model_name='VQVAE',
        epoch=1,
        latent=latent,
        pseudo_latent=pseudo,
    )

    saved_file = tmp_path / 'VQVAE' / 'latent_space_epoch_1.html'
    assert saved_file.is_file()
    assert saved_file.stat().st_size > 0
    mock_show.assert_not_called()


def test_latent_space_event_shown_figure(
    active_experiment: Experiment, tmp_path: pathlib.Path, mocker: MockerFixture
) -> None:
    """When show=True, fig.show() is called."""
    plotly_tracker = AutoencoderPlotly(show=True)
    mocker.patch.object(plotly_tracker, '_get_run_dir', return_value=tmp_path)
    active_experiment.trackers.subscribe(plotly_tracker)
    latent = torch.zeros(2, 3)
    pseudo = torch.ones(2, 3)

    mock_show = mocker.patch.object(go.Figure, 'show')

    LatentSpaceEvent(
        source_name='Test',
        model_name='VQVAE',
        epoch=2,
        latent=latent,
        pseudo_latent=pseudo,
    )

    mock_show.assert_called_once()
