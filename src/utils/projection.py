"""Projection utilities for dimensional reduction."""

import torch

from sklearn.decomposition import PCA

__all__ = ['project_to_3d']


def project_to_3d(latent: torch.Tensor, pseudo_latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Project latent tensors to 3D via PCA if dimension > 3.

    Args:
        latent: Continuous latent points tensor (N, D).
        pseudo_latent: Pseudo-input latent points tensor (M, D).

    Returns:
        Tuple of (latent_3d, pseudo_3d) tensors with shape (N, 3) and (M, 3).
    """
    mu_np = latent.detach().cpu().numpy()
    pseudo_np = pseudo_latent.detach().cpu().numpy()
    if mu_np.shape[-1] > 3:
        pca = PCA(n_components=3)
        mu_proj = pca.fit_transform(mu_np)
        pseudo_proj = pca.transform(pseudo_np)
    else:
        mu_proj = mu_np
        pseudo_proj = pseudo_np

    return torch.from_numpy(mu_proj).float(), torch.from_numpy(pseudo_proj).float()
