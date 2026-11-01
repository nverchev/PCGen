"""Codeword VAE (VAEW) module for secondary latent modeling."""

import abc

from collections.abc import Callable
from typing import Final, override

import torch
import torch.nn as nn

from src.config.experiment import Experiment
from src.data.structures import WOutputs
from src.module.decoders import WDecoder
from src.module.encoders import WEncoder
from src.module.layers import Transpose, reset_parameters_recursive

__all__ = [
    'BaseWAutoEncoder',
    'WAutoEncoder',
]


class BaseWAutoEncoder(nn.Module, abc.ABC):
    """Abstract base of the variational autoencoders of the codewords.

    Attributes:
        z_dim: Dimension of the latent variable.
        dim_codes: Number of codes of a codeword.
        embedding_dim: Dimension of each code.
        n_pseudo_input: Number of pseudo inputs of the VampPrior.
    """

    z_dim: Final[int]
    dim_codes: Final[int]
    embedding_dim: Final[int]
    n_pseudo_input: Final[int]

    def __init__(self) -> None:
        """Initialize."""
        super().__init__()
        cfg = Experiment.get_config()
        cfg_w = cfg.w_autoencoding
        assert cfg_w is not None
        self.z_dim = cfg_w.model.z_dim
        self.dim_codes = cfg.autoencoding.model.dim_codes
        self.embedding_dim = cfg.autoencoding.model.embedding_dim
        self.n_pseudo_input = cfg_w.model.n_pseudo_inputs
        return

    @abc.abstractmethod
    @override
    def forward(self, x: torch.Tensor) -> WOutputs:
        """Reconstruct the codewords ``(B, W)``."""


class WAutoEncoder(BaseWAutoEncoder):
    """Variational autoencoder modeling codeword distributions with VampPrior."""

    to_chunks: nn.Sequential
    encoder: WEncoder
    decoder: WDecoder
    init_pseudo: Final[Callable[[torch.Tensor], torch.Tensor]]
    pseudo_inputs: nn.Parameter
    pseudo_mu: nn.Parameter
    pseudo_log_var: nn.Parameter
    updated: bool

    def __init__(self) -> None:
        """Initialize."""
        super().__init__()
        self.to_chunks = nn.Sequential(nn.Unflatten(1, (self.dim_codes, self.embedding_dim)), Transpose(2, 1))
        self.encoder = WEncoder()
        self.decoder = WDecoder()
        self.init_pseudo = nn.init.normal_
        self.pseudo_inputs = nn.Parameter(torch.empty(self.n_pseudo_input, self.embedding_dim, self.dim_codes))
        self.init_pseudo(self.pseudo_inputs)
        self.pseudo_mu = nn.Parameter(torch.empty(self.n_pseudo_input, self.z_dim))
        self.pseudo_log_var = nn.Parameter(torch.empty(self.n_pseudo_input, self.z_dim))
        self.updated = False
        return

    def update_pseudo_latent(self) -> None:
        """Update pseudo-latent parameters from pseudo inputs."""
        _, _, pseudo_mu, pseudo_log_var = self.encode(None)
        self.pseudo_mu = nn.Parameter(pseudo_mu)
        self.pseudo_log_var = nn.Parameter(pseudo_log_var)
        return

    @override
    def forward(self, x: torch.Tensor) -> WOutputs:
        """Encode and decode codeword inputs."""
        mu, log_var, pseudo_mu, pseudo_log_var = self.encode(x)
        z = self.gaussian_sample(mu, log_var)
        return WOutputs(self.decode(z), z, mu, log_var, pseudo_mu, pseudo_log_var)

    @staticmethod
    def gaussian_sample(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Sample from Gaussian with reparameterization trick."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def encode(self, opt_x: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode input into latent representations.

        Returns:
            The mean and log variance of the latent variable, then those of the pseudo-inputs.
        """
        if opt_x is None:
            x: torch.Tensor = self.pseudo_inputs
        else:
            x = torch.cat([self.to_chunks(opt_x), self.pseudo_inputs], dim=0)

        x = self.encoder(x)
        mu, log_var = x[: -self.n_pseudo_input].chunk(2, 1)
        pseudo_mu, pseudo_log_var = x[-self.n_pseudo_input :].chunk(2, 1)
        return mu, log_var, pseudo_mu, pseudo_log_var

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent z into the reconstructed codeword."""
        return self.decoder(z)

    def reset_parameters(self) -> None:
        """Reset all parameters across submodules."""
        self.init_pseudo(self.pseudo_inputs)
        reset_parameters_recursive(self.encoder)
        reset_parameters_recursive(self.decoder)
        return
