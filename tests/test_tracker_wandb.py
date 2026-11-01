"""Tests for Weights & Biases tracker."""

from pytest_mock import MockerFixture
import torch
from wandb.sdk.wandb_run import Run

from src.config import Experiment
from src.lib.events import GenerationEvent, LatentSpaceEvent, ReconstructionEvent
from src.data.structures import Outputs
from src.trackers.wandb import AutoencoderWandb


def test_wandb_latent_space_event(active_experiment: Experiment, mocker: MockerFixture) -> None:
    """AutoencoderWandb logs 3D object for LatentSpaceEvent."""
    wandb_tracker = AutoencoderWandb()
    active_experiment.trackers.subscribe(wandb_tracker)
    mock_run = mocker.create_autospec(Run, instance=True)
    wandb_tracker._run = mock_run
    latent = torch.zeros(2, 3)
    pseudo = torch.ones(2, 3)

    LatentSpaceEvent(
        source_name='Test',
        model_name='VQVAE',
        epoch=1,
        latent=latent,
        pseudo_latent=pseudo,
    )

    mock_run.define_metric.assert_called_once_with('VQVAE/Test-latent_space', step_metric='Progress/VQVAE')
    mock_run.log.assert_called_once()
    assert 'VQVAE/Test-latent_space' in mock_run.log.call_args[0][0]
    assert mock_run.log.call_args[0][0]['Progress/VQVAE'] == 1
    assert 'step' not in mock_run.log.call_args.kwargs


def test_wandb_reconstruction_event(active_experiment: Experiment, mocker: MockerFixture) -> None:
    """AutoencoderWandb logs reconstructed 3D clouds."""
    wandb_tracker = AutoencoderWandb()
    active_experiment.trackers.subscribe(wandb_tracker)
    mock_run = mocker.create_autospec(Run, instance=True)
    wandb_tracker._run = mock_run
    outputs = Outputs(torch.zeros(2, 10, 3), torch.empty(0), torch.empty(0))

    ReconstructionEvent(outputs=outputs, sample_indices=[0, 1], epoch=3)

    mock_run.log.assert_called_once()
    assert 'reconstruction/recon_0' in mock_run.log.call_args[0][0]
    assert 'reconstruction/recon_1' in mock_run.log.call_args[0][0]
    assert mock_run.log.call_args[0][0]['Progress/autoencoder'] == 3
    assert 'step' not in mock_run.log.call_args.kwargs


def test_wandb_generation_event(active_experiment: Experiment, mocker: MockerFixture) -> None:
    """AutoencoderWandb logs generated 3D clouds."""
    wandb_tracker = AutoencoderWandb()
    active_experiment.trackers.subscribe(wandb_tracker)
    mock_run = mocker.create_autospec(Run, instance=True)
    wandb_tracker._run = mock_run
    outputs = Outputs(torch.zeros(3, 10, 3), torch.empty(0), torch.empty(0))

    GenerationEvent(outputs=outputs, epoch=5)

    mock_run.log.assert_called_once()
    assert 'generation/gen_0' in mock_run.log.call_args[0][0]
    assert 'generation/gen_1' in mock_run.log.call_args[0][0]
    assert 'generation/gen_2' in mock_run.log.call_args[0][0]
    assert mock_run.log.call_args[0][0]['Progress/autoencoder'] == 5
    assert 'step' not in mock_run.log.call_args.kwargs
