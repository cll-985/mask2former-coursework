#!/usr/bin/env python
"""CPU gate test for SDR-v1 head-isolated recovery."""

import importlib.util
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "mask2former" / "modeling" / "detail_recovery.py"
SPEC = importlib.util.spec_from_file_location("detail_recovery", str(MODULE_PATH))
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
Recovery = MODULE.UncertaintyDilationRecovery


def main():
    logits = torch.full((1, 1, 5, 5), -2.0)
    logits[:, :, 1:4, 1:4] = -0.2
    logits[:, :, 2, 2] = 2.0

    recovery = Recovery(
        mask_threshold=0.5,
        uncertainty_margin=0.05,
        dilation_radius=1,
        rescue_heads=2,
    )
    blocked = recovery(logits, num_heads=4)
    allowed_counts = [
        int((~blocked[head, 0]).sum()) for head in range(4)
    ]

    assert blocked.shape == (4, 1, 25)
    assert blocked.dtype == torch.bool
    assert allowed_counts == [9, 9, 1, 1]

    baseline = Recovery(rescue_heads=0)
    baseline_counts = [
        int((~baseline(logits, num_heads=4)[head, 0]).sum())
        for head in range(4)
    ]
    assert baseline_counts == [1, 1, 1, 1]

    print("SDR_V1_HEAD_ISOLATION_GATE_PASS")
    print("allowed_tokens_per_head=9,9,1,1")


if __name__ == "__main__":
    main()
