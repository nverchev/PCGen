"""Quantized autoencoder chained to its codeword prior."""

from typing import override

import numpy as np
import torch
import torch.nn as nn

from src.data.structures import Inputs, Outputs
from src.module.autoencoders import VQVAE
from src.module.w_autoencoders import WAutoEncoder

__all__ = ['VQVAEWithVAE']


class VQVAEWithVAE(nn.Module):
    """Quantized autoencoder decoding the codes its codeword prior chooses.

    Calling it performs the double reconstruction: the autoencoder encodes the
    cloud, the prior encodes and reconstructs that codeword, and the
    autoencoder decodes the codes nearest to the reconstruction. The two parts are trained and saved separately:
    this module only chains them, and holds no weights of its own.

    Attributes:
        autoencoder: Trained autoencoder decoding the codes.
        w_autoencoder: Trained codeword prior choosing the codes.
    """

    autoencoder: VQVAE
    w_autoencoder: WAutoEncoder

    def __init__(self, autoencoder: VQVAE, w_autoencoder: WAutoEncoder, /) -> None:
        """Initialize.

        Args:
            autoencoder: trained autoencoder decoding the codes.
            w_autoencoder: trained codeword prior choosing the codes.
        """
        super().__init__()
        self.autoencoder = autoencoder
        self.w_autoencoder = w_autoencoder
        return

    @override
    def forward(self, inputs: Inputs) -> Outputs:
        """Reconstruct clouds through both encoders: the double reconstruction."""
        w_q = self.autoencoder.encode(inputs.cloud)
        w_recon = self.w_autoencoder(w_q.detach()).w_recon
        return self.autoencoder.decode_nearest_codes(w_recon, inputs.initial_sampling)

    def get_sampling(self, n_clouds: int, /, *, loop: bool) -> torch.Tensor:
        """Get the sampling of the generating surface of the autoencoder for the evaluation.

        Args:
            n_clouds: the number of clouds.
            loop: whether to append a closed path through the corners of the sampling space.

        Returns:
            The sampling, empty when the autoencoder draws none.
        """
        return self.autoencoder.get_sampling(n_clouds, loop=loop)

    @torch.inference_mode()
    def random_sampling(
        self,
        batch_size: int,
        initial_sampling: torch.Tensor | None = None,
        z_bias: torch.Tensor | None = None,
    ) -> Outputs:
        """Sample novel shapes from the prior.

        Args:
            batch_size: shapes to sample.
            initial_sampling: sampling of the generating surface, drawn when
                left out, and moved onto the weights of the module.
            z_bias: shift added to the latent variable of every shape, moved
                onto the weights of the module.

        Returns:
            The sampled shapes and the latent quantities behind them.
        """
        self.eval()
        device = next(self.parameters()).device
        initial_sampling = None if initial_sampling is None else initial_sampling.to(device)
        z_bias = None if z_bias is None else z_bias.to(device)
        pseudo_mu = self.w_autoencoder.pseudo_mu
        pseudo_log_var = self.w_autoencoder.pseudo_log_var
        pseudo_z_list: list[torch.Tensor] = []
        for _ in range(batch_size):
            i = int(np.random.randint(self.w_autoencoder.n_pseudo_input))
            pseudo_z_list.append(self.w_autoencoder.gaussian_sample(pseudo_mu[i], pseudo_log_var[i]))

        pseudo_z = torch.stack(pseudo_z_list).contiguous()
        z = pseudo_z if z_bias is None else pseudo_z + z_bias
        return self.autoencoder.decode_nearest_codes(self.w_autoencoder.decode(z), initial_sampling)
