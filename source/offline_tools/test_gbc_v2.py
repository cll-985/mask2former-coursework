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
)


def square(size, top, left, side):
    mask = torch.zeros(1, 1, size, size)
    mask[:, :, top : top + side, left : left + side] = 1.0
    return mask


def main():
    masks = torch.cat(
        [
            square(64, 2, 2, 3),
            square(64, 8, 8, 12),
            square(64, 12, 12, 40),
        ],
        dim=0,
    )
    legacy = geometry_weights(
        masks,
        small_area_ratio=0.01,
        min_weight=1.0,
        max_weight=2.0,
        normalize=False,
    )
    balanced = geometry_weights(
        masks,
        small_area_ratio=0.01,
        min_weight=0.25,
        max_weight=2.0,
        normalize=True,
    )

    assert torch.all(legacy >= 1.0) and torch.all(legacy <= 2.0)
    assert abs(balanced.mean().item() - 1.0) < 1e-6
    assert balanced[0] > balanced[1] and balanced[0] > balanced[2], balanced
    assert (balanced[0] / balanced[2]) > (legacy[0] / legacy[2])

    target = square(64, 20, 20, 16)
    logits = (target * 12.0 - 6.0).requires_grad_(True)
    loss = geometry_weighted_boundary_loss(
        logits,
        target,
        num_masks=1.0,
        band_width=1,
        dice_weight=1.0,
        bce_weight=0.5,
        small_area_ratio=0.01,
        geo_weight_min=0.25,
        geo_weight_max=2.0,
        normalize_weights=True,
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()

    print("legacy_weights={}".format(legacy.tolist()))
    print("balanced_weights={}".format(balanced.tolist()))
    print("balanced_mean={:.6f}".format(balanced.mean().item()))
    print("GBC_V2_UNIT_GATE_PASS")


if __name__ == "__main__":
    main()
