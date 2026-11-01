from typing import Any

import torch

def ApproxMatch(tensor1: torch.Tensor, tensor2: torch.Tensor) -> tuple[torch.Tensor, Any]: ...
def MatchCost(tensor1: torch.Tensor, tensor2: torch.Tensor, tensor3: torch.Tensor) -> torch.Tensor: ...
def MatchCostGrad(
    tensor1: torch.Tensor, tensor2: torch.Tensor, tensor3: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]: ...
def NNDistance(tensor1: torch.Tensor, tensor2: torch.Tensor) -> torch.Tensor: ...
def NNDistanceGrad(
    tensor1: torch.Tensor,
    tensor2: torch.Tensor,
    indices1: torch.Tensor,
    indices2: torch.Tensor,
    grad1: torch.Tensor,
    grad2: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]: ...
