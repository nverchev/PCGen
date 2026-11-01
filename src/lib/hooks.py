"""Hooks to execute during training."""

import logging

from collections.abc import Sequence
from typing import Any, Final, override

import torch
from torch import distributed as dist

from drytorch.lib import evaluations, hooks, loading, monitoring
from drytorch.lib.training import TrainerBuilder
from drytorch.utils.averages import get_trailing_mean

from src.lib.events import ReconstructionEvent
from src.data import DataSplit
from src.data.structures import Inputs, Outputs, Targets, get_quantization
from src.module import BaseModel, WAutoEncoder, autoencoders

__all__ = [
    'ReconstructionHook',
    'ReviveCodebook',
    'UpdatePseudoLatent',
    'register_checkpointing',
    'register_codebook_revival',
    'register_early_stopping',
    'register_pruning',
    'register_pseudo_latent_update',
    'register_reconstruction_logging',
]


class UpdatePseudoLatent(hooks.TrainerHook[monitoring.Condition]):
    """Update pseudo-latent parameters of the codeword encoder.

    Attributes:
        w_autoencoder: Codeword prior whose pseudo latent is updated.
        when: Decides whether the hook runs when the trainer calls it.
    """

    w_autoencoder: Final[WAutoEncoder]

    def __init__(
        self,
        w_autoencoder: WAutoEncoder,
        /,
        *,
        when: monitoring.Condition = monitoring.ALWAYS,
    ) -> None:
        """Initialize.

        Args:
            w_autoencoder: Codeword prior whose pseudo latent is updated.
            when: Decides whether the hook runs. Default runs every time.
        """
        super().__init__(when=when)
        self.w_autoencoder = w_autoencoder
        return

    @override
    def run(self) -> None:
        """Update pseudo-latent parameters."""
        self.w_autoencoder.update_pseudo_latent()
        return


class ReviveCodebook(hooks.TrainerHook[monitoring.Condition]):
    """Move the codes no sample chose next to a code that is used.

    A code the encoder never picks gets no gradient, so it stays where it was
    initialized and the codebook loses capacity. Moving it onto a code chosen
    often, with noise, gives it a share of that code's samples to claim.

    Attributes:
        evaluation: Pass over the training clouds, collecting the chosen codes.
        module: Quantized head whose codebook is revived.
        noise: Scale of the noise added to the code a dead one is moved onto.
        when: decides whether the hook runs when the trainer calls it.
    """

    diagnostic: Final[evaluations.Evaluation[Inputs, Targets, Outputs]]
    module: Final[autoencoders.VQVAE]
    noise: Final[float]

    def __init__(
        self,
        diagnostic: evaluations.Evaluation[Inputs, Targets, Outputs],
        module: autoencoders.VQVAE,
        /,
        *,
        noise: float,
        when: monitoring.Condition = monitoring.ALWAYS,
    ) -> None:
        """Initialize.

        Args:
            evaluation: pass over the training clouds, which stores the codes
                every sample chose.
            module: quantized head whose codebook is revived.
            noise: scale of the noise added to the code a dead one moves onto.
            when: decides whether the hook runs. Default runs every time.
        """
        super().__init__(when=when)
        self.diagnostic = diagnostic
        self.module = module
        self.noise = noise
        return

    def _is_distributed(self) -> bool:
        return dist.is_available() and dist.is_initialized()

    def _rank(self) -> int:
        return dist.get_rank() if self._is_distributed() else 0

    def _broadcast(self) -> None:
        dist.broadcast(self.module.codebook.data, src=0)

    @override
    def run(self) -> None:
        """Replace every unused code and report how much of the book is in use.

        Side Effects:
            Evaluates the model on the training clouds and writes the codebook.
        """
        self.diagnostic(store_outputs=True)
        is_distributed = self._is_distributed()

        if not is_distributed or self._rank() == 0:
            chosen = torch.vstack([get_quantization(outputs).one_hot_idx for outputs in self.diagnostic.outputs_list])
            counts = chosen.sum(0)
            unused = counts == 0
            logging.info(f'Codebook usage: {1 - unused.float().mean().item():.1%} of the codes.')
            codebook = self.module.codebook.data
            for group in range(counts.shape[0]):
                weights = counts[group] / counts[group].sum()
                for code in torch.nonzero(unused[group]).flatten().tolist():
                    donor = int(torch.multinomial(weights, 1).item())
                    used = codebook[group, donor]
                    codebook[group, code] = used + self.noise * torch.randn_like(used)

        if is_distributed:
            self._broadcast()

        return


def register_checkpointing(builder: TrainerBuilder[Any, Any, Any], /, *, checkpoint_every: int) -> None:
    """Register the checkpointing hook.

    Args:
        builder: trainer builder to register the saving hook to.
        checkpoint_every: save frequency in epochs; ``0`` adds no saving.
    """
    if checkpoint_every:
        builder.add_saving(checkpoint_every)

    return


def register_codebook_revival(
    builder: TrainerBuilder[Any, Any, Any],
    /,
    *,
    diagnostic: evaluations.Evaluation[Inputs, Targets, Outputs],
    module: autoencoders.VQVAE,
    noise: float,
    revive_every: int,
) -> None:
    """Register the codebook revival hook.

    Args:
        builder: trainer builder to register the hook to.
        diagnostic: pass over the training clouds collecting the chosen codes.
        module: quantized head whose codebook is revived.
        noise: scale of the noise added to the code a dead one moves onto.
        revive_every: epochs between two revivals.
    """
    builder.add_post_epoch_hook(
        ReviveCodebook(
            diagnostic,
            module,
            noise=noise,
            when=monitoring.Every(revive_every),
        )
    )
    return


def register_pseudo_latent_update(
    builder: TrainerBuilder[Any, Any, Any],
    w_encoder: WAutoEncoder,
    /,
    *,
    when: monitoring.Condition = monitoring.ALWAYS,
) -> None:
    """Register the pseudo-latent update hook to run after each epoch.

    Args:
        builder: trainer builder to register the hook to.
        w_encoder: codeword encoder whose pseudo latent is updated.
        when: decides whether the hook runs. Default runs every time.
    """
    builder.add_post_epoch_hook(UpdatePseudoLatent(w_encoder, when=when))
    return


def register_early_stopping(builder: TrainerBuilder[Any, Any, Any], /, *, patience: int = 0, window: int) -> None:
    """Register the early stopping hook.

    Args:
        builder: trainer builder to register early stopping to.
        patience: number of evaluations without improvement before stopping.
        window: trailing window size for the smoothed metric.
    """
    builder.add_early_stopping(
        patience=patience,
        watch=builder.watch(filter_fn=get_trailing_mean(window)),
    )
    return


def register_pruning(builder: TrainerBuilder[Any, Any, Any], /, trial: Any = None, *, window: int) -> None:
    """Register the pruning hook.

    Args:
        builder: trainer builder to register pruning callback to.
        trial: Optuna trial object.
        window: number of last validations the pruner averages; ``1`` reads the last one.
    """
    from drytorch.contrib.optuna import TrialCallback

    builder.add_post_epoch_hook(TrialCallback(trial, watch=builder.watch(filter_fn=get_trailing_mean(window))))
    return


class ReconstructionHook(hooks.TrainerHook[monitoring.Condition]):
    """Emit the reconstructions of fixed samples periodically during training.

    Attributes:
        clouds: Point clouds tensor ``(B, N, 3)`` to reconstruct.
        sample_indices: Sample indices corresponding to the clouds.
        loop: Whether the sampling ends with a closed path through the corners of the sampling space.
        when: Decides whether the hook runs when the trainer calls it.
    """

    clouds: Final[torch.Tensor]
    sample_indices: Final[Sequence[int]]
    loop: Final[bool]

    def __init__(
        self,
        clouds: torch.Tensor,
        /,
        *,
        sample_indices: Sequence[int],
        loop: bool,
        when: monitoring.Condition = monitoring.ALWAYS,
    ) -> None:
        """Initialize.

        Args:
            clouds: Point clouds tensor ``(B, N, 3)`` to reconstruct.
            sample_indices: Indices of the samples in the dataset.
            loop: whether the sampling ends with a closed path through the corners of the sampling space.
            when: Decides whether the hook runs. Default runs every time.
        """
        super().__init__(when=when)
        self.clouds = clouds
        self.sample_indices = sample_indices
        self.loop = loop
        return

    @override
    def run(self) -> None:
        """Reconstruct the fixed samples and publish ReconstructionEvent."""
        model = self.trainer.model
        module = model.module
        if not isinstance(module, BaseModel):
            msg = f'Expected BaseModel, got {type(module)}'
            raise TypeError(msg)

        device = model.device
        with torch.inference_mode():
            module.eval()
            sampling = module.get_sampling(self.clouds.shape[0], loop=self.loop)
            inputs = Inputs(cloud=self.clouds.to(device), initial_sampling=sampling)
            outputs = module(inputs)

        ReconstructionEvent(
            outputs=outputs,
            sample_indices=self.sample_indices,
            epoch=model.epoch,
            inputs=inputs,
            is_double=False,
            source_name=model.name,
            model_name=model.name,
        )
        return


def register_reconstruction_logging(
    builder: TrainerBuilder[Any, Any, Any],
    /,
    *,
    n_samples: int,
    split: DataSplit,
    sample_indices: Sequence[int],
    reconstruction_every: int,
    loop: bool,
) -> None:
    """Register the reconstruction logging hook to emit reconstructions periodically.

    Args:
        builder: Trainer builder to register the hook to.
        n_samples: Number of random samples to reconstruct when no indices are given.
        split: Dataset partition to take fixed samples from.
        sample_indices: Indices of samples to reconstruct; empty takes ``n_samples`` random ones.
        reconstruction_every: Number of epochs between reconstruction logging calls.
        loop: whether the sampling ends with a closed path through the corners of the sampling space.

    Raises:
        IndexError: if an index is outside the split.
    """
    if reconstruction_every <= 0:
        return

    if sample_indices:
        taken, (inputs, _) = loading.take_from_dataset(split, indices=sample_indices)
    else:
        taken, (inputs, _) = loading.take_from_dataset(split, n_samples=n_samples, preserve_order=False)

    clouds = inputs.cloud
    builder.add_post_epoch_hook(
        ReconstructionHook(
            clouds,
            sample_indices=taken,
            loop=loop,
            when=monitoring.Every(reconstruction_every),
        )
    )
    return
