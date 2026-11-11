#!/usr/bin/env python3
"""Check reported deltas and recover the primary CI from archived draw results."""
import csv
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "results"


def close(actual, expected, tolerance=1e-9):
    if not math.isclose(actual, expected, rel_tol=0, abs_tol=tolerance):
        raise ValueError(f"Expected {expected}, got {actual}")


def quantile(values, fraction):
    index = (len(values) - 1) * fraction
    left = int(index)
    right = min(left + 1, len(values) - 1)
    return values[left] + (index - left) * (values[right] - values[left])


def main():
    with (ROOT / "primary_comparisons.csv").open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        close(float(row["hard_NMS_AP"]) - float(row["single_AP"]), float(row["delta_AP"]))
    frozen = next(row for row in rows if row["cohort"] == "CarDD original test")
    with (ROOT / "frozen_test_bootstrap_draws.csv").open(newline="", encoding="utf-8-sig") as f:
        draws = list(csv.DictReader(f))
    if len(draws) != 2000 or [int(row["draw"]) for row in draws] != list(range(1, 2001)):
        raise ValueError("Unexpected bootstrap draw roster")
    for row in draws:
        close(float(row["hard_ap"]) - float(row["seed3407_ap"]), float(row["delta"]))
    values = sorted(float(row["delta"]) for row in draws)
    lo, hi = (quantile(values, p) for p in (0.025, 0.975))
    close(lo, float(frozen["CI95_low"]))
    close(hi, float(frozen["CI95_high"]))
    print(f"ARCHIVED_RESULTS_PASS: {len(rows)} AP comparisons, 95% CI [{lo:.6f}, {hi:.6f}]")


if __name__ == "__main__":
    main()
