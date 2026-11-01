"""Model wrappers of the two training phases."""

from typing import Any

import torch

from drytorch.core.exceptions import ModelNotFoundError
from drytorch.lib import models

from src.config import Experiment
from src.data.structures import Inputs, Outputs, WOutputs
from src.module import autoencoders, double_encoders, w_autoencoders
from src.module.generation import Generation, Resampling

__all__ = [
    'VQVAEModel',
    'get_autoencoder_model',
    'get_generation_model',
    'get_vqvae_model',
    'get_vqvae_vae_model',
    'get_w_encoder_model',
    'load_trained_state',
    'load_vqvae_vae',
    'load_vqvae_vae_module',
    'load_w_encoder_model',
]


class VQVAEModel(models.Model[Inputs, Outputs]):
    """drytorch Model wrapper for the quantized autoencoder."""

    module: autoencoders.VQVAE

    @torch.inference_mode()
    def encode(self, clouds: torch.Tensor, /) -> torch.Tensor:
        """Encode clouds under the model's autocast precision.

        Args:
            clouds: input batch of point clouds.

        Returns:
            The unquantized codewords, in float32.
        """
        with torch.autocast(self.device.type, dtype=self.mixed_precision, enabled=self.mixed_precision is not None):
            w_q = self.module.encode(clouds)

        return w_q.float()


def get_autoencoder_model(*, mixed_precision: torch.dtype | None = None) -> models.Model[Inputs, Outputs]:
    """Get the model of the autoencoder phase.

    Args:
        mixed_precision: if not None, the autocast precision.

    Returns:
        The configured head, wrapped for training and checkpointing.
    """
    cfg = Experiment.get_config()
    return models.Model(
        autoencoders.get_autoencoder(),
        device=cfg.user.device,
        mixed_precision=mixed_precision,
        name=cfg.autoencoding.name,
        torch_compile=cfg.user.torch_compile,
    )


def get_vqvae_model(*, mixed_precision: torch.dtype | None = None) -> VQVAEModel:
    """Get the quantized autoencoder model of the autoencoder phase.

    Args:
        mixed_precision: if not None, the autocast precision.

    Returns:
        The configured head, wrapped for encoding codewords.

    Raises:
        ValueError: if the configured head is not the quantized one.
    """
    cfg = Experiment.get_config()
    module = autoencoders.get_autoencoder()
    if not isinstance(module, autoencoders.VQVAE):
        msg = 'Only the quantized head holds the codebook the codewords come from.'
        raise ValueError(msg)

    return VQVAEModel(
        module,
        device=cfg.user.device,
        mixed_precision=mixed_precision,
        name=cfg.autoencoding.name,
        torch_compile=cfg.user.torch_compile,
    )


def get_vqvae_vae_model(module: double_encoders.VQVAEWithVAE, /) -> models.Model[Inputs, Outputs]:
    """Get the model reconstructing clouds through the codeword prior.

    Args:
        module: trained autoencoder and prior, chained.

    Returns:
        The chained models, wrapped for evaluation.
    """
    cfg = Experiment.get_config()
    return models.Model(
        module,
        device=cfg.user.device,
        name='vqvae_vae',
        torch_compile=cfg.user.torch_compile,
    )


def get_generation_model(module: Generation | Resampling, /, *, name: str) -> models.Model[torch.Tensor, torch.Tensor]:
    """Get the model drawing the clouds the generative scores compare.

    Args:
        module: draws one cloud per index.
        name: name of the model.

    Returns:
        The module, wrapped for evaluation.
    """
    cfg = Experiment.get_config()
    return models.Model(
        module,
        device=cfg.user.device,
        name=name,
    )


def get_w_encoder_model() -> models.Model[torch.Tensor, WOutputs]:
    """Get the model of the codeword phase.

    Returns:
        A new codeword prior, wrapped for training and checkpointing.
    """
    cfg = Experiment.get_config()
    assert cfg.w_autoencoding is not None
    return models.Model(
        w_autoencoders.WAutoEncoder(),
        device=cfg.user.device,
        name=cfg.w_autoencoding.name,
        torch_compile=cfg.user.torch_compile,
    )


def load_vqvae_vae(autoencoder: autoencoders.VQVAE, /) -> double_encoders.VQVAEWithVAE:
    """Chain a trained autoencoder to the last prior trained in the run.

    Args:
        autoencoder: trained autoencoder decoding the codes.

    Returns:
        The autoencoder and its prior, chained.

    Raises:
        TypeError: if the wrapped module is not a codeword prior.
    """
    w_autoencoder_module = load_w_encoder_model().module
    if not isinstance(w_autoencoder_module, w_autoencoders.WAutoEncoder):
        msg = 'The loaded module is not a codeword prior.'
        raise TypeError(msg)

    return double_encoders.VQVAEWithVAE(autoencoder, w_autoencoder_module)


def load_vqvae_vae_module() -> double_encoders.VQVAEWithVAE:
    """Chain the trained autoencoder of the run to its last trained prior.

    The autoencoder holds the checkpoint that ``user.load`` selects, the last
    one by default.

    Returns:
        The autoencoder and its prior, chained.

    Raises:
        ValueError: if the configured head is not the quantized one.
    """
    cfg = Experiment.get_config()
    autoencoder = get_autoencoder_model()
    load_trained_state(autoencoder, epoch=cfg.user.load or -1)
    module = autoencoder.module
    if not isinstance(module, autoencoders.VQVAE):
        msg = 'Only the quantized head has a codeword prior.'
        raise ValueError(msg)

    return load_vqvae_vae(module)


def load_w_encoder_model() -> models.Model[torch.Tensor, WOutputs]:
    """Get the codeword prior with the weights of a finished run.

    Returns:
        The model, holding the last checkpoint.
    """
    model = get_w_encoder_model()
    load_trained_state(model)
    return model


def load_trained_state(model: models.Model[Any, Any], /, *, epoch: int = -1) -> None:
    """Load the weights of a model trained by an earlier phase of the experiment.

    Args:
        model: the model to load.
        epoch: the epoch to load. Default loads the last saved epoch.

    Raises:
        RuntimeError: if the experiment holds no trained model.
    """
    try:
        model.load_state(epoch)
    except ModelNotFoundError as error:
        cfg = Experiment.get_config()
        msg = (
            f'No trained {model.name} for the experiment {cfg.name} in {cfg.par_dir}: '
            'train it first with the same configuration and overrides.'
        )
        raise RuntimeError(msg) from error

    return
