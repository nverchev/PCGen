"""Tests for TensorBoard tracker."""

from pytest_mock import MockerFixture
import torch
from torch.utils.tensorboard import SummaryWriter

from src.config import Experiment
from src.lib.events import GenerationEvent, LatentSpaceEvent, ReconstructionEvent
from src.data.structures import Outputs
from src.trackers.tensorboard import AutoencoderTensorBoard


def test_tensorboard_latent_space_event(active_experiment: Experiment, mocker: MockerFixture) -> None:
    """AutoencoderTensorBoard adds mesh for LatentSpaceEvent."""
    tb_tracker = AutoencoderTensorBoard()
    active_experiment.trackers.subscribe(tb_tracker)
    mock_writer = mocker.create_autospec(SummaryWriter, instance=True)
    tb_tracker._writer = mock_writer
    latent = torch.zeros(2, 3)
    pseudo = torch.ones(2, 3)

    LatentSpaceEvent(
        source_name='Test',
        model_name='VQVAE',
        epoch=1,
        latent=latent,
        pseudo_latent=pseudo,
    )

    mock_writer.add_mesh.assert_called_once()
    assert mock_writer.add_mesh.call_args.kwargs['global_step'] == 1
    assert 'VQVAE/Test-latent_space' in mock_writer.add_mesh.call_args[0]


def test_tensorboard_reconstruction_event(active_experiment: Experiment, mocker: MockerFixture) -> None:
    """AutoencoderTensorBoard logs reconstructed clouds."""
    tb_tracker = AutoencoderTensorBoard()
    active_experiment.trackers.subscribe(tb_tracker)
    mock_writer = mocker.create_autospec(SummaryWriter, instance=True)
    tb_tracker._writer = mock_writer
    outputs = Outputs(torch.zeros(2, 10, 3), torch.empty(0), torch.empty(0))

    ReconstructionEvent(outputs=outputs, sample_indices=[0, 1], epoch=3)

    assert mock_writer.add_mesh.call_count == 2


def test_tensorboard_generation_event(active_experiment: Experiment, mocker: MockerFixture) -> None:
    """AutoencoderTensorBoard logs generated clouds."""
    tb_tracker = AutoencoderTensorBoard()
    active_experiment.trackers.subscribe(tb_tracker)
    mock_writer = mocker.create_autospec(SummaryWriter, instance=True)
    tb_tracker._writer = mock_writer
    outputs = Outputs(torch.zeros(3, 10, 3), torch.empty(0), torch.empty(0))

    GenerationEvent(outputs=outputs, epoch=5)

    assert mock_writer.add_mesh.call_count == 3
