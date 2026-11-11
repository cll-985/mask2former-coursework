#!/usr/bin/env python
"""CPU gate test for the parameter-free SDR-v0 attention mask."""

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
    logits[:, :, 2, 2] = 2.0
    # sigmoid(-0.2) ~= 0.45: blocked by the baseline threshold but retained
    # as an uncertain candidate inside the local recovery neighbourhood.
    logits[:, :, 1:4, 1:4] = -0.2
    logits[:, :, 2, 2] = 2.0

    recovery = Recovery(
        mask_threshold=0.5,
        uncertainty_margin=0.15,
        dilation_radius=1,
    )
    blocked = recovery(logits, num_heads=2)
    allowed = ~blocked[0, 0].reshape(5, 5)

    assert blocked.shape == (2, 1, 25)
    assert blocked.dtype == torch.bool
    assert int(allowed.sum()) == 9
    assert bool(allowed[2, 2])
    assert not bool(allowed[0, 0])

    no_recovery = Recovery(dilation_radius=0)
    baseline_allowed = ~no_recovery(logits, num_heads=1)[0, 0].reshape(5, 5)
    assert int(baseline_allowed.sum()) == 1

    print("SDR_V0_UNIT_GATE_PASS")
    print("baseline_allowed=1 sdr_allowed=9 added_tokens=8")


if __name__ == "__main__":
    main()
