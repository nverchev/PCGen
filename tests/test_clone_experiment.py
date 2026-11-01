"""Tests for the clone_experiment entry point."""

import sys

import pytest
from pytest_mock import MockerFixture

from clone_experiment import main


def test_clone_experiment_success(mocker: MockerFixture) -> None:
    """Cloning with valid overrides calls local_ops.clone_experiment_data."""
    mock_clone = mocker.patch('drytorch.utils.local_ops.clone_experiment_data')
    mocker.patch.object(
        sys,
        'argv',
        ['clone_experiment.py', '--config-name', 'vqvae_airplane', 'w_autoencoding.train.n_epochs=1000'],
    )

    main()

    mock_clone.assert_called_once()
    args = mock_clone.call_args[0]
    assert args[1] == 'development'
    assert args[2] == 'development_w_autoencoding.train.n_epochs=1000'


def test_clone_experiment_same_name_error(mocker: MockerFixture) -> None:
    """Cloning when from_name equals destination name raises SystemExit."""
    mocker.patch.object(sys, 'argv', ['clone_experiment.py', '--config-name', 'vqvae_airplane'])

    with pytest.raises(SystemExit):
        main()


def test_clone_experiment_final_requires_from(mocker: MockerFixture) -> None:
    """Cloning a final experiment without --from raises SystemExit."""
    mocker.patch.object(sys, 'argv', ['clone_experiment.py', '--config-name', 'vqvae_airplane', 'final=true'])

    with pytest.raises(SystemExit):
        main()


def test_clone_experiment_final_mismatch_error(mocker: MockerFixture) -> None:
    """Cloning when --from mismatches final status raises SystemExit."""
    mocker.patch.object(
        sys,
        'argv',
        [
            'clone_experiment.py',
            '--config-name',
            'vqvae_airplane',
            '--from',
            'other_final',
            'w_autoencoding.train.n_epochs=1000',
        ],
    )

    with pytest.raises(SystemExit):
        main()
