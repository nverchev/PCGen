"""Tests for the models of the phases."""

import pytest
import torch

from src.lib.models import get_autoencoder_model, get_vqvae_model


@pytest.mark.parametrize('overrides', [['autoencoding.train.mixed_precision=float16', 'user.device=cpu']])
class TestMixedPrecision:
    """Test that only the trained model uses the mixed precision of its phase."""

    def test_trained_autoencoder_uses_mixed_precision(self) -> None:
        """The autoencoder of the training phase uses the precision of the phase."""
        model = get_autoencoder_model(mixed_precision=torch.float16)

        assert model.mixed_precision is torch.float16

    def test_autoencoder_outside_training_uses_full_precision(self) -> None:
        """The autoencoder outside its training phase uses full precision."""
        model = get_autoencoder_model()

        assert model.mixed_precision is None

    def test_vqvae_model_uses_given_mixed_precision(self) -> None:
        """The VQ-VAE model uses the given mixed precision."""
        model = get_vqvae_model(mixed_precision=torch.float16)

        assert model.mixed_precision is torch.float16


@pytest.mark.parametrize('overrides', [['autoencoding.model.model_head=ae', 'user.device=cpu']])
def test_get_vqvae_model_raises_for_non_vqvae() -> None:
    """get_vqvae_model raises ValueError when the configured head is not VQVAE."""
    with pytest.raises(ValueError, match=r'Only the quantized head holds the codebook the codewords come from\.'):
        get_vqvae_model()
