"""Tests for custom layers and batch normalization momentum configuration."""

import pytest
import torch.nn as nn

from src.config.hydra import get_experiment_config
from src.config.torch import get_activation_cls
from src.module.decoders import PCGen
from src.module.encoders import LDGCNN
from src.module.layers import (
    EdgeConvLayer,
    LinearLayer,
    Max,
    Mean,
    PointsConvLayer,
    Repeat,
    Transpose,
)


def test_ae_and_vqvae_configs_have_expected_bn_momentum() -> None:
    """ae.yaml and vqvae.yaml set bn_momentum to 0.1 in architecture."""
    cfg_ae = get_experiment_config(overrides=['autoencoding=autoencoder_ae'])
    cfg_vqvae = get_experiment_config(overrides=['autoencoding=autoencoder_vqvae'])

    assert cfg_ae.autoencoding.architecture.bn_momentum == 0.1
    assert cfg_vqvae.autoencoding.architecture.bn_momentum == 0.1


def test_layers_bn_momentum_default_and_explicit() -> None:
    """Layers default to 0.1; explicit momentum is respected."""
    lin = LinearLayer(10, 20, act_cls=None)
    pts = PointsConvLayer(10, 20, act_cls=None, bn_momentum=0.2)
    edge = EdgeConvLayer(10, 20, act_cls=None, bn_momentum=0.05)

    assert isinstance(lin.norm, nn.BatchNorm1d)
    assert lin.norm.momentum == 0.1
    assert isinstance(pts.norm, nn.BatchNorm1d)
    assert pts.norm.momentum == 0.2
    assert isinstance(edge.norm, nn.BatchNorm2d)
    assert edge.norm.momentum == 0.05


@pytest.mark.parametrize('overrides', [['autoencoding.architecture.bn_momentum=0.2']])
def test_encoder_and_decoder_pass_architecture_bn_momentum() -> None:
    """Encoder and decoder extract bn_momentum from architecture and pass it to layers."""
    encoder = LDGCNN()
    decoder = PCGen()
    edge_layer = encoder.edge_block[0]
    group_block = decoder.group_conv[0]
    group_layer = group_block[0] if isinstance(group_block, nn.Sequential) else None

    assert encoder.bn_momentum == 0.2
    assert isinstance(edge_layer, EdgeConvLayer)
    assert isinstance(edge_layer.norm, nn.BatchNorm2d)
    assert edge_layer.norm.momentum == 0.2
    assert decoder.bn_momentum == 0.2
    assert isinstance(group_block, nn.Sequential)
    assert isinstance(group_layer, PointsConvLayer)
    assert group_layer is not None and isinstance(group_layer.norm, nn.BatchNorm1d)
    assert group_layer is not None and group_layer.norm.momentum == 0.2


def test_the_settings_of_the_shape_modules_appear_in_the_representation() -> None:
    """Every shape module shows its settings in its representation."""
    text = repr(nn.Sequential(Max(dim=3), Mean(dim=-1), Repeat(1, 4), Transpose(2, 1)))

    assert all(setting in text for setting in ('dim=3', 'dim=-1', 'sizes=(1, 4)', 'dim0=2, dim1=1'))


def test_the_sloped_leaky_relu_has_slope_two_tenths() -> None:
    """The default activation of the settings has negative slope 0.2, and initializes as a LeakyReLU."""
    act = get_activation_cls('SlopedLeakyReLU')()
    layer = LinearLayer(4, 4, act_cls=get_activation_cls('SlopedLeakyReLU'), batch_norm=False)

    assert isinstance(act, nn.LeakyReLU)
    assert act.negative_slope == 0.2
    assert isinstance(layer.act, nn.LeakyReLU)
    assert layer.act.negative_slope == 0.2
