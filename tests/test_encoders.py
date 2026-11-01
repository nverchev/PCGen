"""Tests for the PCGen encoders."""

import pytest
import torch

from src.config import Experiment
from src.module.encoders import LDGCNN, WEncoder


def test_representation_shows_every_operation() -> None:
    """The operations of the encoder appear as modules in its representation."""
    text = repr(LDGCNN())

    assert all(name in text for name in ('Transpose', 'NearestNeighbours', 'EdgeFeatures', 'GraphMaxPooling'))


@pytest.mark.parametrize(
    'overrides',
    [[f'w_autoencoding.architecture.encoder_h_dims={h_dims}'] for h_dims in ([4], [8, 16], [16, 8, 4])],
)
def test_w_encoder_h_dims(active_experiment: Experiment) -> None:
    """The encoder maps the codes to the posterior for any number of hidden layers."""
    cfg = active_experiment.config
    x = torch.randn(2, cfg.autoencoding.model.embedding_dim, cfg.autoencoding.model.dim_codes)

    encoded = WEncoder()(x)

    assert cfg.w_autoencoding is not None
    assert encoded.shape == (2, 2 * cfg.w_autoencoding.model.z_dim)
