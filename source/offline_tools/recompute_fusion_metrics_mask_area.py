#!/usr/bin/env python3
"""Recompute saved fusion predictions with correct mask-area size bins."""

from __future__ import print_function

import argparse
import csv
import json
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from offline_tools.fuse_coco_instances import evaluate_results


DEFAULT_ROOTS = (
    "output_offline/P6_instance_consensus_R50",
    "output_offline/P8_baseline_seed_consensus_R50",
    "output_offline/P10_fusion_rule_controls_R50",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--roots", nargs="+", default=list(DEFAULT_ROOTS))
    return parser.parse_args()


def atomic_dump(value, path, indent=None):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temporary = path + ".tmp"
    with open(temporary, "w") as handle:
        json.dump(value, handle, indent=indent, sort_keys=indent is not None)
    os.replace(temporary, path)


def discover(roots):
    paths = []
    for root in roots:
        for directory, _, files in os.walk(root):
            for filename in files:
                # Detectron2 writes ``coco_instances_results.json`` while the
                # fusion tools prepend a rule name.  Both are valid saved
                # prediction files and must use the same mask-area evaluator.
                if filename.endswith("coco_instances_results.json"):
                    paths.append(os.path.join(directory, filename))
    return sorted(paths)


def main():
    args = parse_args()
    paths = discover(args.roots)
    if not paths:
        raise RuntimeError("No saved COCO instance result files found")
    records = []
    detailed = {}
    for index, path in enumerate(paths, 1):
        print("[{}/{}] {}".format(index, len(paths), path), flush=True)
        metrics = evaluate_results(args.gt_json, path)
        detailed[path] = metrics
        for task in ("bbox", "segm"):
            row = {"path": path, "task": task}
            row.update({key: metrics[task][key]
                        for key in ("AP", "AP50", "AP75", "APs", "APm", "APl")})
            records.append(row)
        atomic_dump(
            metrics,
            os.path.join(os.path.dirname(path), "metrics_corrected_mask_area.json"),
            indent=2,
        )

    os.makedirs(args.output_dir, exist_ok=True)
    atomic_dump(
        detailed,
        os.path.join(args.output_dir, "fusion_metrics_corrected_mask_area.json"),
        indent=2,
    )
    csv_path = os.path.join(args.output_dir, "fusion_metrics_corrected_mask_area.csv")
    temporary = csv_path + ".tmp"
    with open(temporary, "w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("path", "task", "AP", "AP50", "AP75", "APs", "APm", "APl"),
        )
        writer.writeheader()
        writer.writerows(records)
    os.replace(temporary, csv_path)
    print("Wrote {} evaluations".format(len(paths)), flush=True)


if __name__ == "__main__":
    main()
