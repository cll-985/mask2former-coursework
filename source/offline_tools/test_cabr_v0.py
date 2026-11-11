#!/usr/bin/env python
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch

from mask2former.modeling.criterion import (
    conflict_aware_boundary_loss,
    project_boundary_gradients,
    soft_high_resolution_boundary,
)


def main():
    # A one-pixel high-resolution object must survive as a soft target after
    # downsampling; this is the failure mode of nearest-first GBC targets.
    thin = torch.zeros(1, 1, 64, 64)
    thin[:, :, 31, 8:56] = 1.0
    soft_boundary = soft_high_resolution_boundary(thin, (16, 16), radius=1)
    assert soft_boundary.sum().item() > 0.0
    assert ((soft_boundary > 0.0) & (soft_boundary < 1.0)).any()

    region = torch.tensor([[[[1.0, 0.0]]]])
    conflicting = torch.tensor([[[[-2.0, 3.0]]]])
    safe, stats = project_boundary_gradients(region, conflicting, norm_cap=0.10)
    dot = (safe * region).sum().item()
    assert dot >= -1e-6, dot
    assert safe.norm().item() <= 0.10 * region.norm().item() + 1e-6
    assert stats["conflict"].item() == 1.0

    aligned = 0.01 * region
    safe_aligned, aligned_stats = project_boundary_gradients(
        region, aligned, norm_cap=0.10
    )
    assert torch.allclose(safe_aligned, aligned, atol=1e-7)
    assert aligned_stats["conflict"].item() == 0.0

    target = torch.zeros(2, 1, 64, 64)
    target[0, :, 15:35, 17:37] = 1.0
    target[1, :, 30, 10:54] = 1.0
    logits = torch.randn(2, 1, 16, 16, requires_grad=True)
    loss, diagnostics = conflict_aware_boundary_loss(
        logits,
        target,
        num_masks=2.0,
        grad_norm_cap=0.10,
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    assert diagnostics["post_alignment"].min().item() >= -1e-6

    if torch.cuda.is_available():
        amp_target = target.cuda()
        amp_logits = torch.randn(2, 1, 16, 16, device="cuda", requires_grad=True)
        with torch.cuda.amp.autocast(enabled=True):
            amp_loss, _ = conflict_aware_boundary_loss(
                amp_logits,
                amp_target,
                num_masks=2.0,
                grad_norm_cap=0.10,
            )
        assert torch.isfinite(amp_loss)
        amp_loss.backward()
        assert amp_logits.grad is not None and torch.isfinite(amp_logits.grad).all()

    empty_logits = torch.zeros(0, 1, 16, 16, requires_grad=True)
    empty_target = torch.zeros(0, 1, 64, 64)
    empty_loss, _ = conflict_aware_boundary_loss(
        empty_logits, empty_target, num_masks=1.0
    )
    assert empty_loss.item() == 0.0

    print("soft_boundary_mass={:.6f}".format(soft_boundary.sum().item()))
    print("conflict_rate={:.6f}".format(diagnostics["conflict"].mean().item()))
    print("cap_rate={:.6f}".format(diagnostics["capped"].mean().item()))
    print("min_post_alignment={:.9f}".format(
        diagnostics["post_alignment"].min().item()
    ))
    print("CABR_V0_UNIT_GATE_PASS")


if __name__ == "__main__":
    main()
