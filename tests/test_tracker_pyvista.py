"""Tests for PyVista tracker."""

import pathlib

import torch
from pytest_mock import MockerFixture

from src.config import Experiment, ExperimentConfig
from src.config.options import Overlays
from src.lib.events import GenerationEvent, ReconstructionEvent
from src.data.structures import Inputs, Outputs
from src.trackers.pyvista import PyVistaTracker


def _get_tracker(cfg: ExperimentConfig) -> PyVistaTracker:
    return PyVistaTracker(
        show=False,
        overlay=cfg.user.plot.overlay,
        sample_indices=cfg.user.plot.sample_indices,
        class_names=[str(name) for name in cfg.data.dataset.select_classes],
        m_test=cfg.autoencoding.train.m_test,
        n_components=cfg.autoencoding.architecture.decoder.n_components,
    )


def test_pyvista_reconstruction_event_notifies(active_experiment: Experiment, mocker: MockerFixture) -> None:
    """PyVistaTracker renders clouds and original clouds separated in folders by index."""
    cfg = active_experiment.config
    tracker = _get_tracker(cfg)
    active_experiment.trackers.subscribe(tracker)
    mock_render = mocker.patch.object(PyVistaTracker, 'render_cloud')
    outputs = Outputs(torch.zeros(2, cfg.autoencoding.train.m_test, 3), torch.empty(0), torch.empty(0))
    inputs = Inputs(cloud=torch.zeros(2, 2048, 3))

    ReconstructionEvent(outputs=outputs, sample_indices=[0, 1], epoch=1, inputs=inputs)

    assert mock_render.call_count == 4
    names = [call.kwargs['name'] for call in mock_render.call_args_list]
    save_dirs = [str(call.kwargs['save_dir']) for call in mock_render.call_args_list]
    assert any('original' in name for name in names)
    assert any('recon' in name and 'epoch_1' in name for name in names)
    assert any(d.endswith('/0') for d in save_dirs)
    assert any(d.endswith('/1') for d in save_dirs)


def test_pyvista_double_reconstruction_event_notifies(active_experiment: Experiment, mocker: MockerFixture) -> None:
    """PyVistaTracker renders double reconstructions with double_recon prefix."""
    cfg = active_experiment.config
    tracker = _get_tracker(cfg)
    active_experiment.trackers.subscribe(tracker)
    mock_render = mocker.patch.object(PyVistaTracker, 'render_cloud')
    outputs = Outputs(torch.zeros(2, cfg.autoencoding.train.m_test, 3), torch.empty(0), torch.empty(0))
    inputs = Inputs(cloud=torch.zeros(2, 2048, 3))

    event = ReconstructionEvent(outputs=outputs, sample_indices=[0, 1], epoch=1, inputs=inputs, is_double=True)

    assert event.is_double is True
    assert mock_render.call_count == 4
    names = [call.kwargs['name'] for call in mock_render.call_args_list]
    save_dirs = [str(call.kwargs['save_dir']) for call in mock_render.call_args_list]
    assert any('original' in name for name in names)
    assert any('double_recon' in name and 'epoch_1' in name for name in names)
    assert any(d.endswith('/0') for d in save_dirs)
    assert any(d.endswith('/1') for d in save_dirs)


def test_pyvista_generation_event_notifies(active_experiment: Experiment, mocker: MockerFixture) -> None:
    """PyVistaTracker renders clouds upon GenerationEvent separated into folders by index."""
    cfg = active_experiment.config
    tracker = _get_tracker(cfg)
    active_experiment.trackers.subscribe(tracker)
    mock_render = mocker.patch.object(PyVistaTracker, 'render_cloud')
    outputs = Outputs(torch.zeros(3, cfg.autoencoding.train.m_test, 3), torch.empty(0), torch.empty(0))

    GenerationEvent(outputs=outputs, epoch=1)

    assert mock_render.call_count == 3
    save_dirs = [str(call.kwargs['save_dir']) for call in mock_render.call_args_list]
    assert any(d.endswith('/0') for d in save_dirs)
    assert any(d.endswith('/1') for d in save_dirs)
    assert any(d.endswith('/2') for d in save_dirs)


def test_pyvista_render_cloud_calls_plotter(mocker: MockerFixture, tmp_path: pathlib.Path) -> None:
    """render_cloud sets up a pyvista Plotter and takes a screenshot offscreen."""
    cloud = torch.zeros(10, 3)
    mock_plotter_cls = mocker.patch('pyvista.Plotter', autospec=True)
    mock_plotter = mock_plotter_cls.return_value

    PyVistaTracker.render_cloud(cloud, name='test_cloud', show=False, save_dir=tmp_path / 'pyvista')

    mock_plotter.screenshot.assert_called_once()
    screenshot_path = mock_plotter.screenshot.call_args[0][0]
    assert 'pyvista' in screenshot_path
    assert 'images' not in screenshot_path
    mock_plotter.close.assert_called_once()


def test_pyvista_resolves_tracker_run_dir(active_experiment: Experiment) -> None:
    """PyVistaTracker resolves output directory inside the tracker run directory."""
    tracker = _get_tracker(active_experiment.config)
    active_experiment.trackers.subscribe(tracker)

    save_dir = tracker._resolve_save_dir()

    assert 'pyvista' in str(save_dir)
    assert save_dir.is_dir()


def test_pyvista_resolves_fallback_dir_outside_scope() -> None:
    """Outside experiment scope, _resolve_save_dir falls back to 'pyvista', not 'images'."""
    tracker = PyVistaTracker(
        show=False, overlay=Overlays.NONE, sample_indices=[], class_names=[], m_test=2048, n_components=4
    )

    save_dir = tracker._resolve_save_dir()

    assert str(save_dir) == 'pyvista'
    assert 'images' not in str(save_dir)
