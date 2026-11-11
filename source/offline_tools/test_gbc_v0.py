#!/usr/bin/env python
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch

from mask2former.modeling.criterion import (
    geometry_weighted_boundary_loss,
    geometry_weights,
    morphological_boundary,
)


def make_square(size, top, left, side):
    mask = torch.zeros(1, 1, size, size)
    mask[:, :, top : top + side, left : left + side] = 1.0
    return mask


def boundary_loss(logits, target):
    return geometry_weighted_boundary_loss(
        logits,
        target,
        num_masks=1.0,
        band_width=1,
        dice_weight=1.0,
        bce_weight=0.5,
        small_area_ratio=0.01,
        geo_weight_max=2.0,
    )


def main():
    target = make_square(64, 20, 20, 16)
    aligned_logits = (target * 12.0 - 6.0).clone().requires_grad_(True)
    shifted_target = torch.roll(target, shifts=(8, 8), dims=(-2, -1))
    shifted_logits = (shifted_target * 12.0 - 6.0).clone().requires_grad_(True)

    aligned_loss = boundary_loss(aligned_logits, target)
    shifted_loss = boundary_loss(shifted_logits, target)
    assert torch.isfinite(aligned_loss)
    assert torch.isfinite(shifted_loss)
    assert aligned_loss.item() < shifted_loss.item(), (aligned_loss, shifted_loss)

    aligned_loss.backward()
    assert aligned_logits.grad is not None
    assert torch.isfinite(aligned_logits.grad).all()

    boundary = morphological_boundary(target, radius=1)
    assert boundary.shape == target.shape
    assert boundary.sum().item() > 0

    small = make_square(64, 4, 4, 3)
    large = make_square(64, 8, 8, 40)
    weights = geometry_weights(
        torch.cat([small, large], dim=0),
        small_area_ratio=0.01,
        max_weight=2.0,
    )
    assert weights.shape == (2,)
    assert 1.0 <= weights.min().item() <= weights.max().item() <= 2.0
    assert weights[0].item() > weights[1].item(), weights

    empty_logits = torch.empty(0, 1, 16, 16, requires_grad=True)
    empty_target = torch.empty(0, 1, 16, 16)
    empty_loss = boundary_loss(empty_logits, empty_target)
    assert empty_loss.item() == 0.0

    print("aligned_loss={:.6f}".format(aligned_loss.item()))
    print("shifted_loss={:.6f}".format(shifted_loss.item()))
    print("geometry_weights={}".format(weights.tolist()))
    print("GBC_V0_UNIT_GATE_PASS")


if __name__ == "__main__":
    main()
