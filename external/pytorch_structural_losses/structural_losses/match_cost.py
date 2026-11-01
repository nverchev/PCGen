from typing import Any, cast

import torch

from torch.autograd import Function

from structural_losses.structural_losses_backend import ApproxMatch, MatchCost, MatchCostGrad


# Inherit from Function
class MatchCostFunction(Function):
    # Note that both forward and backward are static methods
    @staticmethod
    def forward(ctx: Any, *args: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """input:
            set1 : batch_size * #dataset_points * 3
            set2 : batch_size * #query_points * 3
        returns:
            match : batch_size * #query_points * #dataset_points
        """
        set1, set2, *_ = args
        ctx.save_for_backward(set1, set2)
        match, _temp = ApproxMatch(set1, set2)
        ctx.match = match
        cost = MatchCost(set1, set2, match)
        return cost

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> tuple[torch.Tensor, torch.Tensor]:
        grad_output = grad_outputs[0]
        set1, set2 = ctx.saved_tensors
        grad1, grad2 = MatchCostGrad(set1, set2, ctx.match)
        grad_output_expand = grad_output.unsqueeze(1).unsqueeze(2)
        return grad1 * grad_output_expand, grad2 * grad_output_expand

    @classmethod
    def apply(cls, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Apply the function to two tensors."""
        return cast(torch.Tensor, super().apply(x, y))


match_cost = MatchCostFunction.apply
