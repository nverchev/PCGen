"""Tests for loss functions and metrics."""

import pytest
from pytest_mock import MockerFixture
import torch

from src.config import get_experiment_config
from src.data.structures import Outputs, Quantization, Targets, WOutputs
from src.module.autoencoders import VQVAE
from src.lib.objectives import (
    get_autoencoder_loss,
    get_chamfer_loss,
    get_emd_loss,
    get_quantization_loss,
    get_reconstruction_metrics,
    get_w_encoder_loss,
    kullback_leibler,
)


@pytest.fixture
def sample_data() -> tuple[Outputs, Targets]:
    """Return sample outputs and targets for loss testing."""
    cloud1 = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]])
    cloud2 = torch.tensor([[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]])
    outputs = Outputs(cloud2, torch.empty(0), torch.empty(0))
    targets = Targets(ref_cloud=cloud1)
    return outputs, targets


def test_chamfer_loss_computes_distance(sample_data: tuple[Outputs, Targets]) -> None:
    """Chamfer loss is the Chamfer distance averaged over the points of the reference."""
    outputs, targets = sample_data
    loss = get_chamfer_loss()

    result = loss.calculate(outputs, targets)

    assert 'Chamfer' in result
    torch.testing.assert_close(result['Chamfer'], torch.tensor([1.0]))


def test_emd_loss_computes_distance(sample_data: tuple[Outputs, Targets]) -> None:
    """EMD loss is the optimal transport distance averaged over the points of the reference."""
    outputs, targets = sample_data
    loss = get_emd_loss()

    result = loss.calculate(outputs, targets)

    assert 'EMD' in result
    torch.testing.assert_close(result['EMD'], torch.tensor([0.5]))


@pytest.mark.parametrize('overrides', [['autoencoding=autoencoder_ae']])
def test_autoencoder_loss_replicates_today_setup(sample_data: tuple[Outputs, Targets], mocker: MockerFixture) -> None:
    """Default AE config weights Chamfer by the number of points and skips EMD."""
    outputs, targets = sample_data
    mock_emd = mocker.patch('src.lib.objectives.get_emd_loss')
    loss = get_autoencoder_loss()

    result = loss.calculate(outputs, targets)

    mock_emd.assert_not_called()
    assert 'Chamfer' in result
    assert 'EMD' not in result
    torch.testing.assert_close(loss.criterion(result), torch.tensor([2048.0]))


@pytest.mark.parametrize(
    'overrides',
    [['autoencoding=autoencoder_ae', 'autoencoding.train.loss.c_chamfer=0.0', 'autoencoding.train.loss.c_emd=1.0']],
)
def test_autoencoder_loss_skips_chamfer_when_coefficient_is_zero(
    sample_data: tuple[Outputs, Targets], mocker: MockerFixture
) -> None:
    """When c_chamfer is 0, Chamfer loss is not instantiated."""
    outputs, targets = sample_data
    mock_chamfer = mocker.patch('src.lib.objectives.get_chamfer_loss')
    mocker.patch('src.lib.objectives._uses_emd', return_value=True)
    loss = get_autoencoder_loss()

    result = loss.calculate(outputs, targets)

    mock_chamfer.assert_not_called()
    assert 'Chamfer' not in result
    assert 'EMD' in result
    torch.testing.assert_close(loss.criterion(result), torch.tensor([0.5]))


@pytest.mark.parametrize(
    'overrides',
    [['autoencoding=autoencoder_ae', 'autoencoding.train.loss.c_chamfer=1.0', 'autoencoding.train.loss.c_emd=0.0']],
)
def test_autoencoder_loss_skips_emd_when_coefficient_is_zero(
    sample_data: tuple[Outputs, Targets], mocker: MockerFixture
) -> None:
    """When c_emd is 0, EMD loss is not instantiated even if GPU is available."""
    outputs, targets = sample_data
    mock_emd = mocker.patch('src.lib.objectives.get_emd_loss')
    mocker.patch('src.lib.objectives._uses_emd', return_value=True)
    loss = get_autoencoder_loss()

    result = loss.calculate(outputs, targets)

    mock_emd.assert_not_called()
    assert 'EMD' not in result


@pytest.mark.parametrize(
    'overrides',
    [['autoencoding=autoencoder_ae', 'autoencoding.train.loss.c_chamfer=2.0', 'autoencoding.train.loss.c_emd=3.0']],
)
def test_autoencoder_loss_combines_weighted_losses(sample_data: tuple[Outputs, Targets], mocker: MockerFixture) -> None:
    """Both Chamfer and EMD scale according to their respective coefficients."""
    outputs, targets = sample_data
    mocker.patch('src.lib.objectives._uses_emd', return_value=True)
    loss = get_autoencoder_loss()

    result = loss.calculate(outputs, targets)

    torch.testing.assert_close(loss.criterion(result), torch.tensor([3.5]))
    torch.testing.assert_close(result['Chamfer'], torch.tensor([1.0]))
    torch.testing.assert_close(result['EMD'], torch.tensor([0.5]))


@pytest.mark.parametrize(
    'overrides',
    [['autoencoding=autoencoder_ae', 'autoencoding.train.loss.c_chamfer=0.0', 'autoencoding.train.loss.c_emd=0.0']],
)
def test_autoencoder_loss_rejects_all_zero_coefficients() -> None:
    """A ValueError is raised when all loss terms have zero coefficients."""
    with pytest.raises(ValueError, match='positive coefficient'):
        get_autoencoder_loss()


def test_autoencoder_loss_vqvae_replicates_today_setup(
    sample_data: tuple[Outputs, Targets], mocker: MockerFixture
) -> None:
    """VQVAE config replicates today's setup with Chamfer, EMD, and Embed Loss."""
    outputs, targets = sample_data
    outputs.quantization = Quantization(torch.zeros(1, 4), torch.ones(1, 4), torch.empty(0))
    mocker.patch('src.lib.objectives._uses_emd', return_value=True)
    loss = get_autoencoder_loss()

    result = loss.calculate(outputs, targets)

    assert 'Chamfer' in result
    assert 'EMD' in result
    assert 'Embed Loss' in result
    torch.testing.assert_close(result['Chamfer'], torch.tensor([1.0]))
    torch.testing.assert_close(result['EMD'], torch.tensor([0.5]))
    torch.testing.assert_close(result['Embed Loss'], torch.tensor([1.0]))
    torch.testing.assert_close(loss.criterion(result), torch.tensor([3074.0]))


def test_w_encoder_loss_scores_codes_with_autoencoder_distances(mocker: MockerFixture) -> None:
    """The prior is scored on the codes nearest to its reconstruction, measured by the autoencoder."""
    autoencoder = mocker.create_autospec(VQVAE, instance=True)
    autoencoder.square_distances.return_value = torch.tensor([[[4.0, 0.0, 9.0], [1.0, 4.0, 0.0]]])
    outputs = WOutputs(
        torch.zeros(1, 8), torch.zeros(1, 2), torch.zeros(1, 2), torch.zeros(1, 2), torch.zeros(3, 2), torch.zeros(3, 2)
    )
    chosen = torch.tensor([[[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]])
    loss_fn = get_w_encoder_loss(autoencoder)

    result = loss_fn.calculate(outputs, chosen)

    autoencoder.square_distances.assert_called_once_with(outputs.w_recon)
    torch.testing.assert_close(result['Accuracy'], torch.tensor([0.5]))


@pytest.mark.parametrize('overrides', [['autoencoding=autoencoder_ae']])
def test_reconstruction_metrics_include_both_chamfer_and_emd(
    sample_data: tuple[Outputs, Targets], mocker: MockerFixture
) -> None:
    """Reconstruction metrics for final evaluation include both Chamfer and EMD."""
    outputs, targets = sample_data
    mocker.patch('src.lib.objectives._uses_emd', return_value=True)
    metrics = get_reconstruction_metrics()

    result = metrics.calculate(outputs, targets)

    assert 'Chamfer' in result
    assert 'EMD' in result


@pytest.mark.parametrize(
    ('name', 'expected_class', 'expected_commit'),
    [
        ('vqvae_airplane', 'airplane', 2.0),
        ('vqvae_car', 'car', 2.0),
        ('vqvae_chair', 'chair', 2.0),
    ],
)
def test_class_specific_vqvae_configs(name: str, expected_class: str, expected_commit: float) -> None:
    """Class-specific VQVAE configs load correctly with expected class and commitment loss."""
    cfg = get_experiment_config(name)

    assert cfg.par_dir == cfg.user.path.version_dir / name
    assert [c.value for c in cfg.data.dataset.select_classes] == [expected_class]
    assert cfg.autoencoding.train.loss.c_commitment == expected_commit


@pytest.mark.parametrize(
    ('name', 'expected_class'),
    [
        ('ae_airplane', 'airplane'),
        ('ae_car', 'car'),
        ('ae_chair', 'chair'),
    ],
)
def test_class_specific_ae_configs(name: str, expected_class: str) -> None:
    """Class-specific AE configs load correctly without quantization (c_commitment=0, head=ae)."""
    cfg = get_experiment_config(name)

    assert cfg.par_dir == cfg.user.path.version_dir / name
    assert [c.value for c in cfg.data.dataset.select_classes] == [expected_class]
    assert cfg.autoencoding.model.model_head.value == 'ae'
    assert cfg.autoencoding.train.loss.c_commitment == 0.0
    assert cfg.autoencoding.train.loss.c_embedding == 0.0
    assert cfg.autoencoding.architecture.decoder.n_components == 4
    assert cfg.autoencoding.train.n_epochs == 1000


def test_kullback_leibler_returns_float32_for_float16_inputs() -> None:
    """kullback_leibler computes divergence in float32 for float16 outputs."""
    outputs = WOutputs(
        torch.empty(0),
        torch.randn(2, 16, dtype=torch.float16),
        torch.randn(2, 16, dtype=torch.float16),
        torch.randn(2, 16, dtype=torch.float16),
        torch.randn(10, 16, dtype=torch.float16),
        torch.randn(10, 16, dtype=torch.float16),
    )

    kld = kullback_leibler(outputs)

    assert kld.dtype == torch.float32


def test_losses_return_float32_for_float16_inputs(sample_data: tuple[Outputs, Targets]) -> None:
    """Autoencoder loss terms and metrics calculate in float32 for float16 outputs."""
    outputs, orig_targets = sample_data
    outputs.recon = outputs.recon.half()
    targets = Targets(ref_cloud=orig_targets.ref_cloud.half())
    w_q = torch.randn(1, 64, dtype=torch.float16)
    outputs.quantization = Quantization(w_q, torch.randn(1, 64, dtype=torch.float16), torch.empty(0))

    chamfer_loss = get_chamfer_loss().calculate(outputs, targets)
    emd_loss = get_emd_loss().calculate(outputs, targets)
    quant_loss = get_quantization_loss().calculate(outputs, targets)

    assert chamfer_loss['Chamfer'].dtype == torch.float32
    assert emd_loss['EMD'].dtype == torch.float32
    for name, value in quant_loss.items():
        assert value.dtype == torch.float32, f'{name} is not float32'


def test_w_encoder_loss_returns_float32_for_float16_inputs(mocker: MockerFixture) -> None:
    """W-encoder loss calculates all dictionary metrics in float32 for float16 inputs."""
    autoencoder = mocker.create_autospec(VQVAE, instance=True)
    autoencoder.square_distances.return_value = torch.ones(2, 4, 8, dtype=torch.float32)
    outputs = WOutputs(
        torch.randn(2, 64, dtype=torch.float16),
        torch.randn(2, 16, dtype=torch.float16),
        torch.randn(2, 16, dtype=torch.float16),
        torch.randn(2, 16, dtype=torch.float16),
        torch.randn(5, 16, dtype=torch.float16),
        torch.randn(5, 16, dtype=torch.float16),
    )
    one_hot = torch.zeros(2, 4, 8, dtype=torch.float16)
    one_hot[:, :, 0] = 1.0

    loss = get_w_encoder_loss(autoencoder)
    result = loss.calculate(outputs, one_hot)

    for name, value in result.items():
        assert value.dtype == torch.float32, f'{name} is not float32'
