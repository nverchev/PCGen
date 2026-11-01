"""Typed I/O structures for the model, loss and metrics.

Every drytorch project defines:
  ``Inputs``  — ``NamedTuple`` of tensors fed to the model.
  ``Targets`` — ``NamedTuple`` of tensors consumed by the loss.
  ``Outputs`` — dataclass of tensors produced by the model; here one for each autoencoder.
"""

import dataclasses

from typing import Final, NamedTuple

import torch

__all__ = [
    'IN_CHAN',
    'OUT_CHAN',
    'Inputs',
    'Outputs',
    'Quantization',
    'Targets',
    'WOutputs',
    'get_quantization',
]

IN_CHAN: Final = 3
"""Coordinates of a point in an input cloud."""

OUT_CHAN: Final = 3
"""Coordinates of a point in a generated cloud."""


# ---------------------------------------------------------------------------
# Inputs: tensors consumed by the model's forward() method
# ---------------------------------------------------------------------------


class Inputs(NamedTuple):
    """Point cloud fed to the encoder.

    Attributes:
        cloud: Vertex coordinates ``(B, N, 3)``.
        initial_sampling: Sampling of the generating surface ``(B, M, D)``,
            empty unless a fixed sampling is given.
    """

    cloud: torch.Tensor
    initial_sampling: torch.Tensor = torch.empty(0)


# ---------------------------------------------------------------------------
# Targets: tensors consumed by the loss and evaluation metrics
# ---------------------------------------------------------------------------


class Targets(NamedTuple):
    """Reference cloud and the scale that denormalizes it.

    Attributes:
        ref_cloud: Reference coordinates ``(B, N, 3)``, a second sampling of
            the same shape when resampling is on.
        scale: Factor restoring the original coordinates ``(B,)``.
    """

    ref_cloud: torch.Tensor
    scale: torch.Tensor = torch.ones(1)


# ---------------------------------------------------------------------------
# Outputs: whatever the model produces
# ---------------------------------------------------------------------------


@dataclasses.dataclass(slots=True)
class Quantization:
    """Codewords of the quantized head.

    Attributes:
        w_q: Codeword before quantisation.
        w_e: Quantised codeword.
        one_hot_idx: Indicator of the chosen codes, summed into the usage.
    """

    w_q: torch.Tensor
    w_e: torch.Tensor
    one_hot_idx: torch.Tensor


@dataclasses.dataclass(slots=True)
class Outputs:
    """Outputs of the cloud autoencoders.

    Attributes:
        recon: Reconstructed coordinates ``(B, M, 3)``.
        attention: Share of every reconstructed point each component claims
            ``(B, M, C)``, empty unless the decoder mixes several components.
        components: Coordinates every component decodes ``(B, 3, M, C)``,
            empty unless the decoder mixes several components.
        quantization: Codewords of the quantized head, ``None`` for the other heads.
    """

    recon: torch.Tensor
    attention: torch.Tensor
    components: torch.Tensor
    quantization: Quantization | None = None


@dataclasses.dataclass(slots=True)
class WOutputs:
    """Outputs of the codeword autoencoder.

    Attributes:
        w_recon: Reconstructed codeword.
        z: Sample of the continuous latent variable.
        mu: Mean of the continuous latent variable.
        log_var: Log variance of the continuous latent variable.
        pseudo_mu: Means of the pseudo-inputs of the prior.
        pseudo_log_var: Log variances of the pseudo-inputs of the prior.
    """

    w_recon: torch.Tensor
    z: torch.Tensor
    mu: torch.Tensor
    log_var: torch.Tensor
    pseudo_mu: torch.Tensor
    pseudo_log_var: torch.Tensor


def get_quantization(outputs: Outputs, /) -> Quantization:
    """Get the codewords of the quantized head.

    Args:
        outputs: outputs of the quantized head.

    Returns:
        The codewords before and after quantisation, with the chosen codes.

    Raises:
        ValueError: if the outputs come from a head without quantization.
    """
    if outputs.quantization is None:
        raise ValueError('The outputs come from a head without quantization.')

    return outputs.quantization
