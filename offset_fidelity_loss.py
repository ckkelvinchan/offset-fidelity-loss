import torch
import torch.nn as nn


class OffsetFidelityLoss(nn.Module):
    """Offset-Fidelity Loss

    Paper:
        Understanding Deformable Alignment in Video Super-Resolution,
        AAAI, 2021

    Args:
        loss_weight (float): The loss weight of the loss. The variable
            '\lambda' in Eq.(5) of the paper. Default: 1.0.
        threshold (float): The threshold below which the loss is not computed.
            The variable 't' in Eq.(5) of the paper. Default: 10.0.
    """

    def __init__(self, loss_weight=1.0, threshold=10.0):

        super().__init__()

        self.loss_weight = loss_weight
        self.threshold = threshold

    def forward(self, offset, flow):
        """Forward function.

        Given the offset and optical flow, the optical flow is flipped and
        repeated (since optical flow is arranged in (x, y) order and the offset
        is arranged in (y1, x1, y2, x2, ...) order). A mask is then computed
        for the masked L1 Loss. Values smaller than the threshold are masked.

        Args:
            offset (Tensor): The learnable DCN offsets with size (n, c, h, w).
            flow (Tensor): The precomputed optical flow with size (n, 2, h, w).
        """

        n, c, h, w = offset.size()
        offset = offset.view(-1, 2, h, w)  # separate offset in batch dimension

        # flip and repeat the optical flow
        flow = flow.flip(1).repeat(1, c // 2, 1, 1).view(-1, 2, h, w)

        # compute loss
        abs_diff = torch.abs(offset - flow)
        mask = (abs_diff > self.threshold).type_as(abs_diff)
        loss = torch.sum(torch.mean(mask * abs_diff, dim=(1, 2, 3)))

        return self.loss_weight * loss
