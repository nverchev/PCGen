# Structural losses extension

CUDA implementation of the approximate Earth Mover's Distance used as a
baseline in the point cloud generation literature.

The extension computes the earth mover distance of the loss. Without it, the distance falls back to
an exact matching with `scipy`, which is too slow for training.

## Building

The build compiles against `torch`, so `nvcc` must come from the CUDA toolkit that PyTorch was
compiled against (check with `nvcc --version`). From the project root:

```bash
uv sync --extra emd
```

## Example use

`match_cost` returns the matching cost summed over the points, so dividing by
their number gives the per-point distance that `src.utils.distances` reports:

```python
import torch
from structural_losses import match_cost

cloud1 = torch.rand(2, 2048, 3, device='cuda')
cloud2 = torch.rand(2, 2048, 3, device='cuda')
emd = match_cost(cloud1.contiguous(), cloud2.contiguous()) / cloud1.shape[1]
```
