"""Sparse detail-recovery attention-mask utilities.

The module is deliberately parameter-free. It preserves Mask2Former's
high-confidence mask support and only re-opens uncertain tokens in a small
dilated neighbourhood. SDR-v1 can isolate recovery to a subset of attention
heads so the remaining heads retain the baseline semantic context.
"""

import torch
from torch import nn
from torch.nn import functional as F


class UncertaintyDilationRecovery(nn.Module):
    """Build a local recovery mask from predicted mask logits.

    Input:
        mask_logits: Tensor[B, Q, H, W].
        num_heads: number of decoder attention heads.

    Output:
        BoolTensor[B * num_heads, Q, H * W], where True means blocked.

    The allowed set is ``core union (dilate(core) intersect uncertainty)``.
    The operation is deterministic, parameter-free, and detached from the
    training graph, matching the baseline attention-mask behaviour.
    """

    def __init__(
        self,
        mask_threshold=0.5,
        uncertainty_margin=0.15,
        dilation_radius=1,
        rescue_heads=-1,
    ):
        super().__init__()
        if not 0.0 < mask_threshold < 1.0:
            raise ValueError("mask_threshold must be in (0, 1)")
        if not 0.0 <= uncertainty_margin <= 0.5:
            raise ValueError("uncertainty_margin must be in [0, 0.5]")
        if dilation_radius < 0:
            raise ValueError("dilation_radius must be non-negative")
        if rescue_heads < -1:
            raise ValueError("rescue_heads must be -1 or non-negative")

        self.mask_threshold = float(mask_threshold)
        self.uncertainty_margin = float(uncertainty_margin)
        self.dilation_radius = int(dilation_radius)
        self.rescue_heads = int(rescue_heads)

    def forward(self, mask_logits, num_heads):
        if mask_logits.dim() != 4:
            raise ValueError("mask_logits must have shape [B, Q, H, W]")
        if num_heads < 1:
            raise ValueError("num_heads must be positive")
        if self.rescue_heads > num_heads:
            raise ValueError("rescue_heads cannot exceed num_heads")

        probabilities = mask_logits.detach().sigmoid()
        core = probabilities >= self.mask_threshold

        if self.dilation_radius == 0 or self.uncertainty_margin == 0.0:
            allowed = core
        else:
            kernel_size = 2 * self.dilation_radius + 1
            local_neighbourhood = F.max_pool2d(
                core.float(),
                kernel_size=kernel_size,
                stride=1,
                padding=self.dilation_radius,
            ).bool()
            uncertain = (
                (probabilities - self.mask_threshold).abs()
                <= self.uncertainty_margin
            )
            allowed = core | (local_neighbourhood & uncertain)

        recovered_blocked = ~allowed.flatten(2)
        baseline_blocked = ~core.flatten(2)
        rescue_heads = (
            num_heads if self.rescue_heads == -1 else self.rescue_heads
        )

        # The first rescue_heads receive the locally expanded mask. All other
        # heads keep the exact baseline mask. Concatenation avoids an in-place
        # update and makes the head isolation explicit and deterministic.
        masks = []
        if rescue_heads:
            masks.append(
                recovered_blocked.unsqueeze(1).repeat(
                    1, rescue_heads, 1, 1
                )
            )
        if rescue_heads < num_heads:
            masks.append(
                baseline_blocked.unsqueeze(1).repeat(
                    1, num_heads - rescue_heads, 1, 1
                )
            )
        return torch.cat(masks, dim=1).flatten(0, 1)
