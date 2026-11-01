"""Tests for the autoencoder chained to its codeword prior."""

from unittest.mock import create_autospec

import torch

from src.data.structures import Inputs, Outputs, WOutputs
from src.module import VQVAE, VQVAEWithVAE, WAutoEncoder


def test_vqvae_vae_reconstructs_through_prior() -> None:
    """The autoencoder decodes the codes nearest to the codeword the prior reconstructs."""
    autoencoder = create_autospec(VQVAE, instance=True)
    w_autoencoder = create_autospec(WAutoEncoder, instance=True)
    w_q = torch.randn(2, 8, requires_grad=True)
    autoencoder.encode.return_value = w_q
    w_recon = torch.zeros(2, 8)
    empty = torch.empty(0)
    w_autoencoder.return_value = WOutputs(w_recon, empty, empty, empty, empty, empty)
    expected_outputs = Outputs(empty, empty, empty)
    autoencoder.decode_nearest_codes.return_value = expected_outputs
    inputs = Inputs(cloud=torch.zeros(2, 10, 3), initial_sampling=torch.zeros(2, 5, 3))
    outputs = VQVAEWithVAE(autoencoder, w_autoencoder)(inputs)

    assert outputs is expected_outputs
    assert not w_autoencoder.call_args.args[0].requires_grad
    autoencoder.decode_nearest_codes.assert_called_once_with(w_recon, inputs.initial_sampling)
    return
