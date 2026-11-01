from typing import Any

from torch.autograd import Function

from structural_losses.structural_losses_backend import NNDistance, NNDistanceGrad


# Inherit from Function
class NNDistanceFunction(Function):
    # Note that both forward and backward are static methods
    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any) -> Any:
        """input:
            set1 : batch_size * #dataset_points * 3
            set2 : batch_size * #query_points * 3
        returns:
            dist1, idx1, dist2, idx2
        """
        set1, set2, *_ = args
        ctx.save_for_backward(set1, set2)

        dist1, idx1, dist2, idx2 = NNDistance(set1, set2)
        ctx.idx1 = idx1
        ctx.idx2 = idx2
        return dist1, dist2

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        set1, set2 = ctx.saved_tensors
        idx1 = ctx.idx1
        idx2 = ctx.idx2
        grad1, grad2 = NNDistanceGrad(
            set1, set2, idx1, idx2, grad_outputs[0].contiguous(), grad_outputs[1].contiguous()
        )
        return grad1, grad2


nn_distance = NNDistanceFunction.apply
