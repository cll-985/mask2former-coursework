#!/usr/bin/env python3
"""Recompute frozen-test segmentation AP from deidentified matching events."""
import gzip
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1] / "results"


def average_precision(event):
    labels = np.frombuffer(event["events"].encode("ascii"), dtype=np.uint8) == ord("1")
    positives = event["positives"]
    if positives == 0:
        return None
    if labels.size == 0:
        return 0.0
    tp = np.cumsum(labels)
    fp = np.cumsum(~labels)
    recall = tp / positives
    precision = tp / (tp + fp + np.spacing(1))
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    indices = np.searchsorted(recall, np.linspace(0, 1, 101), side="left")
    valid = indices < labels.size
    samples = np.zeros(101)
    samples[valid] = precision[indices[valid]]
    return float(samples.mean())


def main():
    with gzip.open(ROOT / "frozen_test_matching_events.json.gz", "rt", encoding="utf-8") as f:
        systems = json.load(f)
    result = {}
    for name, system in systems.items():
        ap = [average_precision(event) for event in system["groups"]]
        result[name] = 100 * float(np.mean([x for x in ap if x is not None]))
        if abs(result[name] - system["archived_ap"]) > 1e-9:
            raise ValueError(f"{name}: AP mismatch")
        print(f"{name}: {result[name]:.6f}")
    print(f"AP_EVENT_REPLAY_PASS: gain {result['hard'] - result['seed3407']:+.6f}")


if __name__ == "__main__":
    main()
