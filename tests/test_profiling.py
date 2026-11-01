"""Tests for model profiling wrappers and entry points."""

import pytest
from pytest_mock import MockerFixture
import torch

import profile_autoencoder
from src.config import Experiment
from src.data.structures import Inputs, Outputs
from src.module import AE, Oracle, VQVAE, VQVAEWithVAE
from src.module.generation import Drawing


def test_drawing_forwards_to_random_sampling(mocker: MockerFixture) -> None:
    """Drawing wrapper passes batch_size to random_sampling."""
    mock_module = mocker.create_autospec(VQVAEWithVAE, instance=True)
    expected_outputs = Outputs(torch.empty(0), torch.empty(0), torch.empty(0))
    mock_module.random_sampling.return_value = expected_outputs
    drawing = Drawing(mock_module)

    outputs = drawing(4)

    assert outputs is expected_outputs
    mock_module.random_sampling.assert_called_once_with(4, None)


def test_profile_autoencoder_profiles_ae_reconstruction(active_experiment: Experiment, mocker: MockerFixture) -> None:
    """Configuring an AE profiles encoding and decoding a cloud."""
    cfg = active_experiment.config
    mock_ae = mocker.create_autospec(AE, instance=True)
    mock_ae.eval.return_value = mock_ae
    sampling = torch.zeros(cfg.autoencoding.train.batch_size, 2, 4)
    mock_ae.get_sampling.return_value = sampling
    mock_model = mocker.MagicMock()
    mock_model.device = torch.device('cpu')
    mock_model.module = mock_ae
    mocker.patch('profile_autoencoder.get_autoencoder_model', return_value=mock_model)
    mock_profile = mocker.patch('profile_autoencoder.get_model_profile')
    mocker.patch('profile_autoencoder._time_passes', return_value=[1.0, 2.0, 3.0])

    profile_autoencoder.profile_autoencoder()

    mock_profile.assert_called_once()
    model_arg = mock_profile.call_args.kwargs['model']
    args = mock_profile.call_args.kwargs['args']
    assert model_arg is mock_ae
    assert len(args) == 1
    assert isinstance(args[0], Inputs)
    assert args[0].cloud.shape == (cfg.autoencoding.train.batch_size, cfg.data.dataset.input_points, 3)
    assert args[0].initial_sampling is sampling
    mock_ae.get_sampling.assert_called_once_with(cfg.autoencoding.train.batch_size, loop=False)


def test_profile_autoencoder_profiles_vqvae_drawing(active_experiment: Experiment, mocker: MockerFixture) -> None:
    """Configuring a VQVAE profiles drawing through a new prior."""
    cfg = active_experiment.config
    mock_vqvae = mocker.create_autospec(VQVAE, instance=True)
    mock_vqvae.eval.return_value = mock_vqvae
    sampling = torch.zeros(cfg.autoencoding.train.batch_size, 2, 4)
    mock_vqvae.get_sampling.return_value = sampling
    mock_prior = mocker.MagicMock()
    mock_model = mocker.MagicMock()
    mock_model.device = torch.device('cpu')
    mock_model.module = mock_vqvae
    mocker.patch('profile_autoencoder.get_autoencoder_model', return_value=mock_model)
    mock_prior_cls = mocker.patch('profile_autoencoder.WAutoEncoder')
    mock_prior_cls.return_value.to.return_value = mock_prior
    mock_profile = mocker.patch('profile_autoencoder.get_model_profile')
    mocker.patch('profile_autoencoder._time_passes', return_value=[1.0, 2.0, 3.0])

    profile_autoencoder.profile_autoencoder()

    mock_profile.assert_called_once()
    model_arg = mock_profile.call_args.kwargs['model']
    args = mock_profile.call_args.kwargs['args']
    assert isinstance(model_arg, Drawing)
    assert model_arg.module.autoencoder is mock_vqvae
    assert model_arg.module.w_autoencoder is mock_prior
    mock_prior_cls.assert_called_once_with()
    assert args[0] == cfg.autoencoding.train.batch_size
    assert args[1] is sampling


def test_the_latency_is_timed_after_the_warm_up(mocker: MockerFixture) -> None:
    """The model runs the warm-up passes, then one timed pass for each latency."""
    model = mocker.MagicMock()

    latencies = profile_autoencoder._time_passes(model, [4], torch.device('cpu'))

    assert len(latencies) == profile_autoencoder.N_TIMED_PASSES
    assert model.call_count == profile_autoencoder.N_WARM_UP_PASSES + profile_autoencoder.N_TIMED_PASSES
    model.assert_called_with(4)


def test_profile_autoencoder_rejects_unsupported_head(mocker: MockerFixture) -> None:
    """An unsupported module type raises a ValueError."""
    mock_oracle = mocker.create_autospec(Oracle, instance=True)
    mock_oracle.eval.return_value = mock_oracle
    mock_model = mocker.MagicMock()
    mock_model.device = torch.device('cpu')
    mock_model.module = mock_oracle
    mocker.patch('profile_autoencoder.get_autoencoder_model', return_value=mock_model)

    with pytest.raises(ValueError, match='does not support profiling'):
        profile_autoencoder.profile_autoencoder()
