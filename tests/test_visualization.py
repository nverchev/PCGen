"""Tests for visualization helper functions."""

from unittest.mock import create_autospec, patch

import torch

from src.data.structures import Outputs
from src.module import BaseModel, VQVAEWithVAE
from src.utils.visualization import generate_and_publish, reconstruct_and_publish


def test_reconstruct_and_publish_creates_event() -> None:
    """reconstruct_and_publish runs forward pass and publishes ReconstructionEvent."""
    mock_module = create_autospec(BaseModel, instance=True)
    mock_module.parameters.return_value = iter([torch.zeros(1)])
    expected_outputs = Outputs(torch.zeros(2, 10, 3), torch.empty(0), torch.empty(0))
    mock_module.return_value = expected_outputs

    clouds = torch.zeros(2, 10, 3)
    with (
        patch('src.utils.visualization.ReconstructionEvent') as mock_event,
    ):
        reconstruct_and_publish(mock_module, clouds, sample_indices=[0, 1], epoch=1, loop=False)
        mock_event.assert_called_once()
        call_kwargs = mock_event.call_args.kwargs
        assert call_kwargs['outputs'] is expected_outputs
        assert call_kwargs['sample_indices'] == [0, 1]
        assert call_kwargs['epoch'] == 1
        assert call_kwargs['is_double'] is False

    return


def test_reconstruct_and_publish_with_is_double() -> None:
    """reconstruct_and_publish respects is_double parameter."""
    mock_module = create_autospec(BaseModel, instance=True)
    mock_module.parameters.return_value = iter([torch.zeros(1)])
    expected_outputs = Outputs(torch.zeros(2, 10, 3), torch.empty(0), torch.empty(0))
    mock_module.return_value = expected_outputs

    clouds = torch.zeros(2, 10, 3)
    with (
        patch('src.utils.visualization.ReconstructionEvent') as mock_event,
    ):
        reconstruct_and_publish(mock_module, clouds, sample_indices=[0, 1], epoch=1, loop=False, is_double=True)
        mock_event.assert_called_once()
        call_kwargs = mock_event.call_args.kwargs
        assert call_kwargs['is_double'] is True
        assert call_kwargs['source_name'] == 'double_reconstruction'

    return


def test_generate_and_publish_creates_event() -> None:
    """generate_and_publish runs random_sampling and publishes GenerationEvent."""
    mock_module = create_autospec(VQVAEWithVAE, instance=True)
    mock_module.parameters.return_value = iter([torch.zeros(1)])
    expected_outputs = Outputs(torch.zeros(3, 10, 3), torch.empty(0), torch.empty(0))
    mock_module.random_sampling.return_value = expected_outputs

    z_bias = torch.zeros(3, 16)
    with (
        patch('src.utils.visualization.GenerationEvent') as mock_event,
    ):
        generate_and_publish(mock_module, n_clouds=3, z_bias=z_bias, epoch=1, loop=False)
        mock_event.assert_called_once()
        call_kwargs = mock_event.call_args.kwargs
        assert call_kwargs['outputs'] is expected_outputs
        assert call_kwargs['epoch'] == 1

    return
