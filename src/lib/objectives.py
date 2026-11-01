"""Losses and metrics of the two training phases."""

from __future__ import annotations

import math
import warnings

import torch

from torch import nn
from torch.nn import functional

from drytorch.lib import objectives

from src.config import Experiment
from src.config.options import ModelHeads
from src.data.structures import Outputs, Targets, WOutputs, get_quantization
from src.module import VQVAE
from src.utils import distances

__all__ = [
    'gaussian_log_likelihood',
    'get_autoencoder_loss',
    'get_chamfer_loss',
    'get_emd_loss',
    'get_quantization_loss',
    'get_reconstruction_metrics',
    'get_w_encoder_loss',
    'kullback_leibler',
]


def gaussian_log_likelihood(x: torch.Tensor, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    """Return the log likelihood of a sample, up to a constant.

    Args:
        x: sample.
        mu: mean of the distribution.
        log_var: log variance of the distribution.

    Returns:
        The log likelihood of every coordinate of the sample.
    """
    x, mu, log_var = x.float(), mu.float(), log_var.float()
    return -0.5 * (log_var + (x - mu) ** 2 / log_var.exp())


def kullback_leibler(outputs: WOutputs, /) -> torch.Tensor:
    """Return the divergence of the posterior from the pseudo-input prior.

    The prior is a uniform mixture of the pseudo-input distributions, so the
    divergence is the posterior likelihood minus the mixture likelihood.

    Args:
        outputs: outputs holding the latent sample and both distributions.

    Returns:
        The divergence of every sample of the batch.
    """
    z = outputs.z.float()
    mu = outputs.mu.float()
    log_var = outputs.log_var.float()
    pseudo_mu = outputs.pseudo_mu.float()
    pseudo_log_var = outputs.pseudo_log_var.float()
    posterior = gaussian_log_likelihood(z, mu, log_var).sum(1)
    n_pseudo = pseudo_mu.shape[0]
    batch = mu.shape[0]
    z_exp = z.unsqueeze(1).expand(-1, n_pseudo, -1)
    pseudo_mu_exp = pseudo_mu.unsqueeze(0).expand(batch, -1, -1)
    pseudo_log_var_exp = pseudo_log_var.unsqueeze(0).expand(batch, -1, -1)
    prior = torch.logsumexp(gaussian_log_likelihood(z_exp, pseudo_mu_exp, pseudo_log_var_exp).sum(2), dim=1)
    return posterior - prior + math.log(n_pseudo)


def get_chamfer_loss() -> objectives.Loss[Outputs, Targets]:
    """Get the reconstruction loss based on the Chamfer distance, averaged over the points."""

    def _chamfer_loss(outputs: Outputs, targets: Targets) -> torch.Tensor:
        return distances.chamfer(targets.ref_cloud.float(), outputs.recon.float()) / targets.ref_cloud.shape[1]

    return objectives.Loss(_chamfer_loss, name='Chamfer')


def get_emd_loss() -> objectives.Loss[Outputs, Targets]:
    """Get the reconstruction loss based on the earth mover distance, averaged over the points."""

    def _emd_loss(outputs: Outputs, targets: Targets) -> torch.Tensor:
        return distances.emd(targets.ref_cloud.float(), outputs.recon.float()) / targets.ref_cloud.shape[1]

    return objectives.Loss(_emd_loss, name='EMD')


def get_quantization_loss() -> objectives.LossWithMetrics[Outputs, Targets]:
    """Get the regularization loss of the quantized head."""
    cfg_loss = Experiment.get_config().autoencoding.train.loss
    c_commitment = cfg_loss.c_commitment
    c_embedding = cfg_loss.c_embedding
    mean_square = nn.MSELoss(reduction='none')
    if c_embedding == c_commitment:

        def _embed_loss(outputs: Outputs, _: Targets) -> torch.Tensor:
            quantization = get_quantization(outputs)
            return mean_square(quantization.w_q.float(), quantization.w_e.float()).mean(dim=1)

        embed_loss = objectives.Loss(_embed_loss, name='Embed Loss')
        return c_commitment * embed_loss

    def calculate(outputs: Outputs, _: Targets) -> dict[str, torch.Tensor]:
        quantization = get_quantization(outputs)
        w_q = quantization.w_q.float()
        w_e = quantization.w_e.float()
        commitment = mean_square(w_q, w_e.detach()).mean(dim=1)
        embedding = mean_square(w_q.detach(), w_e).mean(dim=1)
        regularization = c_commitment * commitment + c_embedding * embedding
        return {'Regularization': regularization, 'Embed Loss': commitment}

    return objectives.DictLoss(calculate, name='Regularization')


def get_autoencoder_loss() -> objectives.LossWithMetrics[Outputs, Targets]:
    """Get the reconstruction loss of the autoencoder phase.

    A quantized head adds the commitment and embedding terms, which coincide
    when their two coefficients do.

    Returns:
        The loss, reporting the terms it sums as metrics of their own.

    Side Effects:
        Warns when the configuration asks for the earth mover distance on a run
        that is not on the accelerator, where it is left out.
    """
    cfg = Experiment.get_config()
    cfg_loss = cfg.autoencoding.train.loss
    c_chamfer = cfg_loss.c_chamfer
    c_emd = cfg_loss.c_emd
    quantized = cfg.autoencoding.model.model_head == ModelHeads.VQVAE
    loss: objectives.LossWithMetrics[Outputs, Targets]
    if c_emd > 0 and _uses_emd(True):
        emd = get_emd_loss()
        loss = c_emd * emd
        if c_chamfer > 0:
            chamfer = get_chamfer_loss()
            loss = loss + c_chamfer * chamfer

    elif c_chamfer == 0:
        raise ValueError('At least one loss term must have a positive coefficient.')
    else:
        chamfer = get_chamfer_loss()
        loss = c_chamfer * chamfer

    if quantized:
        loss = loss + get_quantization_loss()

    return loss


def get_reconstruction_metrics() -> objectives.MetricCollection[Outputs, Targets]:
    """Get the distances between a reconstruction and its reference cloud.

    Clouds of equal size are compared per point, so the summed distances are
    divided by the number of points.

    Returns:
        The Chamfer distance and, on the accelerator, the earth mover distance,
        named after the coordinates they are measured on.

    Side Effects:
        Warns when the earth mover distance is left out.
    """
    de_normalize = Experiment.get_config().autoencoding.train.de_normalize
    with_emd = _uses_emd(True)

    def calculate(outputs: Outputs, targets: Targets) -> dict[str, torch.Tensor]:
        recon = outputs.recon.float()
        ref_cloud = targets.ref_cloud.float()
        if de_normalize:
            scale = targets.scale.float().view(-1, 1, 1)
            recon = recon * scale
            ref_cloud = ref_cloud * scale

        values = {'Chamfer': distances.chamfer(ref_cloud, recon)}
        if with_emd:
            values['EMD'] = distances.emd(ref_cloud, recon)

        if ref_cloud.shape == recon.shape:
            values = {name: value / ref_cloud.shape[1] for name, value in values.items()}

        if de_normalize:
            values = {f'Denorm {name}': value for name, value in values.items()}

        return values

    return objectives.MetricCollection(calculate)


def get_w_encoder_loss(autoencoder: VQVAE, /) -> objectives.DictLoss[WOutputs, torch.Tensor]:
    """Get the loss of the codeword phase.

    The codeword encoder predicts which code of every group the autoencoder
    chose, so the loss is the likelihood of those codes plus the divergence of
    the latent variable.

    Args:
        autoencoder: trained autoencoder whose codebook scores the predictions.

    Returns:
        The loss, reporting the divergence, the likelihood and the share of
        codes it predicts correctly.
    """
    cfg_w = Experiment.get_config().w_autoencoding
    assert cfg_w is not None
    c_kld = cfg_w.train.loss.c_kld

    def calculate(outputs: WOutputs, one_hot_idx: torch.Tensor) -> dict[str, torch.Tensor]:
        one_hot_idx = one_hot_idx.float()
        divergence = kullback_leibler(outputs)
        square_distance = autoencoder.square_distances(outputs.w_recon.float())
        distance = torch.sqrt(square_distance)
        scores = -distance + distance.min(2, keepdim=True)[0]
        likelihood = -(scores.log_softmax(dim=2) * one_hot_idx).sum((1, 2))
        predicted = functional.one_hot(square_distance.argmin(2), num_classes=one_hot_idx.shape[2]).float()
        accuracy = (one_hot_idx * predicted).sum(2).mean(1)
        return {
            'Criterion': likelihood + c_kld * divergence,
            'KLD': divergence,
            'NLL': likelihood,
            'Accuracy': accuracy,
        }

    return objectives.DictLoss(calculate, name='Criterion')


def _uses_emd(requested: bool) -> bool:
    # the exact matching that stands in for the extension is too slow for an epoch
    if not requested:
        return False

    device = Experiment.get_config().user.device
    on_accelerator = torch.accelerator.is_available() if device == 'auto' else device.type != 'cpu'
    if on_accelerator:
        return True

    warnings.warn('The earth mover distance needs the accelerator: leaving it out.', stacklevel=3)
    return False
