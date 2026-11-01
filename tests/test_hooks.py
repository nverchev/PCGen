"""Tests for training hooks."""

from unittest.mock import MagicMock, create_autospec, patch

import torch

from src.data import DataSplit
from src.data.structures import Inputs, Outputs, Quantization, Targets
from src.module import BaseModel
from src.lib.hooks import (
    ReconstructionHook,
    ReviveCodebook,
    UpdatePseudoLatent,
    register_codebook_revival,
    register_pseudo_latent_update,
    register_reconstruction_logging,
)


def test_update_pseudo_latent_hook() -> None:
    """UpdatePseudoLatent calls update_pseudo_latent on the w_encoder."""
    mock_w_encoder = MagicMock()
    hook = UpdatePseudoLatent(mock_w_encoder)
    hook.run()
    mock_w_encoder.update_pseudo_latent.assert_called_once()

    mock_builder = MagicMock()
    register_pseudo_latent_update(mock_builder, mock_w_encoder)
    mock_builder.add_post_epoch_hook.assert_called_once()
    return


def test_reconstruction_hook_runs_and_publishes() -> None:
    """ReconstructionHook calls reconstruct_and_publish."""
    mock_model = MagicMock()
    mock_model.module = create_autospec(BaseModel, instance=True)
    mock_model.name = 'TestModel'
    mock_model.epoch = 2
    mock_model.device = torch.device('cpu')
    mock_trainer = MagicMock()
    mock_trainer.model = mock_model

    clouds = torch.zeros(2, 10, 3)
    hook = ReconstructionHook(clouds, sample_indices=[1, 5], loop=False)
    hook.bind_trainer(mock_trainer)

    with (
        patch('src.lib.hooks.ReconstructionEvent') as mock_event,
    ):
        hook.run()
        mock_event.assert_called_once()
        call_kwargs = mock_event.call_args.kwargs
        assert call_kwargs['sample_indices'] == [1, 5]
        assert call_kwargs['source_name'] == 'TestModel'
        assert call_kwargs['epoch'] == 2

    return


def test_register_reconstruction_logging() -> None:
    """register_reconstruction_logging adds ReconstructionHook with Every condition."""
    mock_builder = MagicMock()
    mock_split = create_autospec(DataSplit, instance=True)
    mock_split.__len__.return_value = 5
    mock_split.__getitem__.side_effect = lambda idx: (
        Inputs(cloud=torch.zeros(10, 3)),
        Targets(ref_cloud=torch.zeros(10, 3), scale=torch.tensor(1.0)),
    )
    register_reconstruction_logging(
        mock_builder,
        n_samples=4,
        split=mock_split,
        sample_indices=[0, 2],
        reconstruction_every=5,
        loop=False,
    )
    mock_builder.add_post_epoch_hook.assert_called_once()
    hook_arg = mock_builder.add_post_epoch_hook.call_args[0][0]
    assert isinstance(hook_arg, ReconstructionHook)
    assert hook_arg.clouds.shape == (2, 10, 3)
    return


def test_revive_codebook_runs() -> None:
    """ReviveCodebook updates unused codes in codebook."""
    mock_diagnostic = MagicMock()
    # 1 sample, dim_codes=2, book_size=3
    # Group 0: code 0 used (1, 0, 0)
    # Group 1: code 1 used (0, 1, 0)
    one_hot_idx = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
    empty = torch.empty(0)
    output = Outputs(empty, empty, empty, Quantization(empty, empty, one_hot_idx))
    mock_diagnostic.outputs_list = [output]

    mock_module = MagicMock()
    codebook_tensor = torch.zeros(2, 3, 4)
    mock_module.codebook = torch.nn.Parameter(codebook_tensor)

    hook = ReviveCodebook(mock_diagnostic, mock_module, noise=0.1)
    hook.run()

    mock_diagnostic.assert_called_once_with(store_outputs=True)
    return


def test_revive_codebook_distributed_worker() -> None:
    """ReviveCodebook on non-zero rank does not crash with empty outputs and broadcasts."""
    mock_diagnostic = MagicMock()
    mock_diagnostic.outputs_list = []  # rank != 0 has empty outputs_list

    mock_module = MagicMock()
    mock_module.codebook = torch.nn.Parameter(torch.zeros(2, 3, 4))

    hook = ReviveCodebook(mock_diagnostic, mock_module, noise=0.1)

    with (
        patch.object(hook, '_is_distributed', return_value=True),
        patch.object(hook, '_rank', return_value=1),
        patch.object(hook, '_broadcast') as mock_broadcast,
    ):
        hook.run()
        mock_broadcast.assert_called_once()

    return


def test_register_codebook_revival() -> None:
    """register_codebook_revival adds ReviveCodebook hook to trainer builder."""
    mock_builder = MagicMock()
    mock_diagnostic = MagicMock()
    mock_module = MagicMock()
    register_codebook_revival(
        mock_builder,
        diagnostic=mock_diagnostic,
        module=mock_module,
        noise=0.1,
        revive_every=10,
    )
    mock_builder.add_post_epoch_hook.assert_called_once()
    hook_arg = mock_builder.add_post_epoch_hook.call_args[0][0]
    assert isinstance(hook_arg, ReviveCodebook)
    return
