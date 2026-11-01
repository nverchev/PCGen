"""Tests for derived datasets (codewords dataset)."""

import subprocess
import sys
from unittest.mock import create_autospec

import pytest
import torch
from torch.utils import data

from src.data import DataSplit, WDataset
from src.data.structures import Inputs, Targets
from src.module.autoencoders import VQVAE
from src.lib import VQVAEModel, get_vqvae_model


def test_no_circular_import_loop() -> None:
    """Verifies that importing src.data and src.module in any order does not cause a dependency loop."""
    commands = [
        'import src.data',
        'import src.module',
        'import src.data; import src.module',
        'import src.module; import src.data',
        'from src.data import WDataset',
    ]
    for cmd in commands:
        result = subprocess.run(
            [sys.executable, '-c', cmd],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, f'Command failed: {cmd}\nStderr: {result.stderr}'


def test_w_dataset_encoding() -> None:
    """WDataset correctly delegates encoding to the autoencoder."""
    mock_module = create_autospec(VQVAE, instance=True)
    mock_module.modules.return_value = []
    w_q = torch.randn(2, 4, 8)
    mock_module.quantise.return_value = (w_q, torch.zeros(2, 4, 16))
    mock_model = create_autospec(VQVAEModel, instance=True)
    mock_model.device = torch.device('cpu')
    mock_model.module = mock_module
    mock_model.encode.return_value = w_q

    cloud1 = torch.randn(10, 3)
    cloud2 = torch.randn(10, 3)
    samples = [
        (Inputs(cloud=cloud1), Targets(ref_cloud=cloud1)),
        (Inputs(cloud=cloud2), Targets(ref_cloud=cloud2)),
    ]
    mock_split = create_autospec(DataSplit, instance=True)
    mock_split.__len__.return_value = len(samples)
    mock_split.__getitem__.side_effect = lambda idx: samples[idx]
    dataset = WDataset(mock_split, mock_model)
    items = dataset.__getitems__([0, 1])

    assert len(dataset) == 2
    assert len(items) == 2
    assert mock_model.encode.called
    assert mock_module.quantise.called


def test_w_dataset_requires_sized_split() -> None:
    """WDataset raises TypeError if split is not Sized."""
    mock_model = create_autospec(VQVAEModel, instance=True)
    mock_model.device = torch.device('cpu')
    unsized_split = create_autospec(data.Dataset, instance=True)
    del unsized_split.__len__

    with pytest.raises(TypeError, match=r'The codewords need a split of known length\.'):
        WDataset(unsized_split, mock_model)


@pytest.mark.parametrize('overrides', [['user.device=cpu']])
def test_w_dataset_encoding_with_bfloat16_model_returns_float32() -> None:
    """Encoding with a bfloat16 model returns float32 codewords."""
    model = get_vqvae_model(mixed_precision=torch.bfloat16)
    cloud = torch.randn(64, 3)
    sample = (Inputs(cloud=cloud), Targets(ref_cloud=cloud))
    mock_split = create_autospec(DataSplit, instance=True)
    mock_split.__len__.return_value = 1
    mock_split.__getitem__.side_effect = lambda idx: sample
    dataset = WDataset(mock_split, model)
    items = dataset.__getitems__([0])

    assert len(items) == 1
    w_q, _ = items[0]
    assert w_q.dtype == torch.float32
