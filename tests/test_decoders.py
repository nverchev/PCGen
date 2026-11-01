"""Tests for the PCGen decoder."""

import pytest
import torch

from src.config import Experiment
from src.module.decoders import PCGen, WDecoder


def test_attention_is_deterministic_share_in_evaluation() -> None:
    """Evaluation drops the Gumbel noise, so the same input gives the same shares, which sum to one."""
    decoder = PCGen().eval()
    if decoder.attention is None:
        raise AssertionError('Decoder attention must not be None.')
    features = torch.randn(2, decoder.attention.score.in_dim, 64)

    first = decoder.attention(features)
    second = decoder.attention(features)

    torch.testing.assert_close(first, second)
    torch.testing.assert_close(first.sum(2), torch.ones(2, 64))


def test_representation_shows_every_operation() -> None:
    """The operations of the decoder appear as modules in its representation."""
    text = repr(PCGen())

    assert all(name in text for name in ('SphereProjection', 'Modulate', 'ComponentAttention', 'GraphFilter'))


def test_the_sampling_ends_with_the_loop_through_the_corners_when_asked() -> None:
    """The loop adds points after the random ones, and passes through every corner of the sampling space."""
    decoder = PCGen()

    plain = decoder.draw_sampling(2, 10, loop=False)
    looped = decoder.draw_sampling(2, 10, loop=True)

    corners = torch.cat([torch.eye(decoder.sample_dim), -torch.eye(decoder.sample_dim)], dim=1)
    assert plain.shape == (2, decoder.sample_dim, 10)
    assert looped.shape[:2] == (2, decoder.sample_dim)
    assert looped.shape[2] > 10
    assert all((looped[0, :, 10:] == corner.unsqueeze(1)).all(dim=0).any() for corner in corners.T)


@pytest.mark.parametrize(
    'overrides',
    [[f'w_autoencoding.architecture.decoder_h_dims={h_dims}'] for h_dims in ([32], [128, 64], [64, 32, 16])],
)
def test_w_decoder_h_dims(active_experiment: Experiment) -> None:
    """The decoder maps continuous latent to codeword representation for any hidden layers."""
    cfg = active_experiment.config
    assert cfg.w_autoencoding is not None
    z = torch.randn(2, cfg.w_autoencoding.model.z_dim)

    decoded = WDecoder()(z)

    assert decoded.shape == (2, cfg.autoencoding.model.w_dim)
