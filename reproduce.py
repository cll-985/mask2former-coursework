#!/usr/bin/env python3
"""Recompute the principal coursework results from the bundled evaluation records."""

import bisect
import csv
import gzip
import json
import math
import sys
from pathlib import Path

from reproduction.check_reported_results import close, quantile


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def interpolated_ap(group):
    positives = group["positives"]
    if not positives:
        return None
    labels = group["events"]
    recall, precision = [], []
    true_positives = 0
    for rank, event in enumerate(labels, start=1):
        true_positives += event == "1"
        recall.append(true_positives / positives)
        precision.append(true_positives / (rank + sys.float_info.epsilon))
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])
    indices = (bisect.bisect_left(recall, point * 0.01) for point in range(101))
    return math.fsum(precision[i] if i < len(precision) else 0 for i in indices) / 101


def frozen_test():
    with gzip.open(RESULTS / "frozen_test_matching_events.json.gz", "rt", encoding="utf-8") as f:
        systems = json.load(f)
    values = {}
    for name, entry in systems.items():
        groups = [interpolated_ap(group) for group in entry["groups"]]
        valid_groups = [value for value in groups if value is not None]
        values[name] = 100 * math.fsum(valid_groups) / len(valid_groups)
        close(values[name], entry["archived_ap"])
    frozen = next(row for row in read_csv(RESULTS / "primary_comparisons.csv")
                  if row["cohort"] == "CarDD original test")
    close(values["seed3407"], float(frozen["single_AP"]))
    close(values["hard"], float(frozen["hard_NMS_AP"]))
    values["hard_minus_seed3407"] = values["hard"] - values["seed3407"]
    close(values["hard_minus_seed3407"], float(frozen["delta_AP"]))
    draws = read_csv(RESULTS / "frozen_test_bootstrap_draws.csv")
    if len(draws) != 2000:
        raise ValueError("Expected 2,000 bootstrap draws")
    differences = sorted(float(row["hard_ap"]) - float(row["seed3407_ap"]) for row in draws)
    values["ci95"] = [quantile(differences, p) for p in (0.025, 0.975)]
    for bound, key in zip(values["ci95"], ("CI95_low", "CI95_high")):
        close(bound, float(frozen[key]))
    return values


def calibration():
    configs = {}
    for path in sorted((RESULTS / "calibration_counts").glob("*.csv")):
        rows = read_csv(path)
        for row in rows:
            tp, gt, fp, images = (int(row[key]) for key in ("tp", "gt", "fp", "images"))
            close(100 * tp / gt, float(row["recall_percent"]))
            close(fp / images, float(row["fp_per_image"]))
        primary = {row["system"]: row for row in rows if row["group"] == "all"
                   and float(row["calibration_budget"]) == 1 and float(row["mask_iou"]) == 0.75}
        hard, linear = primary["hard"], primary["linear"]
        if any(hard[key] != linear[key] for key in ("tp", "fp", "gt", "images")):
            raise ValueError(f"Hard and linear calibration counts differ for {path.stem}")
        configs[path.stem] = {
            "images": int(hard["images"]), "ground_truth": int(hard["gt"]),
            "hard_and_linear_tp": int(hard["tp"]), "hard_and_linear_fp": int(hard["fp"]),
            "recall_percent": 100 * int(hard["tp"]) / int(hard["gt"]),
            "fp_per_image": int(hard["fp"]) / int(hard["images"]),
        }
    if len(configs) != 8:
        raise ValueError("Expected eight calibration configurations")
    return configs


def geometry():
    families = {}
    for row in read_csv(RESULTS / "geometry_editor_diagnostics.csv"):
        ap = float(row["AP"])
        close(ap - float(row["reference_AP"]), float(row["delta_AP"]))
        families.setdefault(row["family"], {})[row["policy"]] = ap
    output = {}
    for family, policies in families.items():
        output[family] = {
            "component_minus_whole": policies["component"] - policies["whole"],
            "region_minus_learned_whole": policies["Direct region"] - policies["Learned whole"],
        }
    return output


def main():
    report = {
        "frozen_test_ap_recomputed_from_matching_events": frozen_test(),
        "fp_budget_one_recomputed_from_archived_counts": calibration(),
        "geometry_ap_differences_from_archived_ap": geometry(),
    }
    output = ROOT / "reproduced_results.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    values = report["frozen_test_ap_recomputed_from_matching_events"]
    print(f"CarDD baseline AP: {values['seed3407']:.6f}")
    print(f"CarDD fusion AP:   {values['hard']:.6f}")
    print(f"AP gain: {values['hard_minus_seed3407']:+.6f}")
    print(f"95% interval: [{values['ci95'][0]:.6f}, {values['ci95'][1]:.6f}]")
    print("REPRODUCTION_PASS", output.name)


if __name__ == "__main__":
    main()
