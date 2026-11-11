#!/usr/bin/env python3
"""Paired image bootstrap for saved COCO mask predictions.

The primary paper metric remains dataset-level COCO AP.  This script adds a
paired robustness statistic that is decomposable by image: COCO-style mask AP
is computed independently per image (IoU 0.50:0.95, 101 recall points), then
candidate-minus-baseline image deltas are bootstrapped.
"""

from __future__ import print_function

import argparse
import csv
import json
import os
import sys
from collections import defaultdict

import numpy as np
from pycocotools import mask as mask_util
from pycocotools.coco import COCO

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from offline_tools.fuse_coco_instances import evaluate_results, rle_for_compute


IOU_THRESHOLDS = np.arange(0.50, 0.96, 0.05)
RECALL_THRESHOLDS = np.linspace(0.0, 1.0, 101)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-json", required=True)
    parser.add_argument("--comparison", action="append", required=True,
                        help="name::baseline.json::candidate.json")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261008)
    return parser.parse_args()


def parse_comparison(value):
    fields = value.split("::")
    if len(fields) != 3 or not all(fields):
        raise ValueError("comparison must be name::baseline.json::candidate.json")
    return fields


def atomic_dump(value, path, indent=None):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temporary = path + ".tmp"
    with open(temporary, "w") as handle:
        json.dump(value, handle, indent=indent, sort_keys=indent is not None)
    os.replace(temporary, path)


def load_predictions(path):
    with open(path, "r") as handle:
        values = json.load(handle)
    grouped = defaultdict(list)
    for item in values:
        grouped[(int(item["image_id"]), int(item["category_id"]))].append(item)
    for key in grouped:
        grouped[key].sort(key=lambda item: -float(item["score"]))
    return grouped


def load_ground_truth(coco):
    grouped = defaultdict(list)
    for annotation in coco.dataset["annotations"]:
        if int(annotation.get("iscrowd", 0)):
            continue
        grouped[(int(annotation["image_id"]), int(annotation["category_id"]))].append(
            coco.annToRLE(annotation))
    return grouped


def interpolated_ap(tp, fp, ground_truth_count):
    if ground_truth_count <= 0:
        return 0.0
    tp = np.cumsum(np.asarray(tp, dtype=np.float64))
    fp = np.cumsum(np.asarray(fp, dtype=np.float64))
    recall = tp / float(ground_truth_count)
    precision = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    values = []
    for threshold in RECALL_THRESHOLDS:
        valid = precision[recall >= threshold]
        values.append(float(np.max(valid)) if valid.size else 0.0)
    return float(np.mean(values))


def category_ap(predictions, ground_truth):
    if not ground_truth:
        return 0.0 if predictions else None
    if not predictions:
        return 0.0
    prediction_rles = [rle_for_compute(item["segmentation"]) for item in predictions]
    overlaps = mask_util.iou(
        prediction_rles,
        ground_truth,
        [0] * len(ground_truth),
    )
    threshold_aps = []
    for threshold in IOU_THRESHOLDS:
        matched = set()
        tp = []
        fp = []
        for prediction_index in range(len(predictions)):
            best_index = -1
            best_iou = float(threshold)
            for gt_index in range(len(ground_truth)):
                if gt_index in matched:
                    continue
                value = float(overlaps[prediction_index, gt_index])
                if value >= best_iou:
                    best_iou = value
                    best_index = gt_index
            if best_index >= 0:
                matched.add(best_index)
                tp.append(1.0)
                fp.append(0.0)
            else:
                tp.append(0.0)
                fp.append(1.0)
        threshold_aps.append(interpolated_ap(tp, fp, len(ground_truth)))
    return float(np.mean(threshold_aps))


def image_scores(coco, gt_grouped, prediction_grouped):
    category_ids = [int(value) for value in coco.getCatIds()]
    output = {}
    for image_id in sorted(int(value) for value in coco.getImgIds()):
        values = []
        for category_id in category_ids:
            gt = gt_grouped.get((image_id, category_id), [])
            predictions = prediction_grouped.get((image_id, category_id), [])
            value = category_ap(predictions, gt)
            if value is not None:
                values.append(value)
        output[image_id] = 100.0 * float(np.mean(values)) if values else 0.0
    return output


def bootstrap_delta(deltas, resamples, seed):
    rng = np.random.RandomState(seed)
    deltas = np.asarray(deltas, dtype=np.float64)
    count = len(deltas)
    means = np.empty(resamples, dtype=np.float64)
    chunk = 1000
    for start in range(0, resamples, chunk):
        size = min(chunk, resamples - start)
        indices = rng.randint(0, count, size=(size, count))
        means[start:start + size] = np.mean(deltas[indices], axis=1)
    return {
        "mean_delta": float(np.mean(deltas)),
        "median_delta": float(np.median(deltas)),
        "ci95_low": float(np.percentile(means, 2.5)),
        "ci95_high": float(np.percentile(means, 97.5)),
        "bootstrap_probability_positive": float(np.mean(means > 0.0)),
        "image_win_rate": float(np.mean(deltas > 0.0)),
        "image_tie_rate": float(np.mean(deltas == 0.0)),
        "images": int(count),
    }


def main():
    args = parse_args()
    if args.resamples <= 0:
        raise ValueError("resamples must be positive")
    comparisons = [parse_comparison(value) for value in args.comparison]
    coco = COCO(args.gt_json)
    gt_grouped = load_ground_truth(coco)
    unique_paths = sorted(set(path for _, base, candidate in comparisons
                              for path in (base, candidate)))
    score_cache = {}
    metric_cache = {}
    for index, path in enumerate(unique_paths, 1):
        print("score [{}/{}] {}".format(index, len(unique_paths), path), flush=True)
        score_cache[path] = image_scores(coco, gt_grouped, load_predictions(path))
        metric_cache[path] = evaluate_results(args.gt_json, path)["segm"]

    report = {
        "protocol": {
            "metric": "per-image COCO-style mask AP",
            "iou_thresholds": [float(value) for value in IOU_THRESHOLDS],
            "recall_points": 101,
            "resamples": args.resamples,
            "seed": args.seed,
            "note": "Dataset-level COCO AP remains the primary metric; bootstrap uses a paired image-decomposable robustness statistic.",
        },
        "comparisons": {},
    }
    rows = []
    image_ids = sorted(int(value) for value in coco.getImgIds())
    for offset, (name, baseline_path, candidate_path) in enumerate(comparisons):
        baseline = score_cache[baseline_path]
        candidate = score_cache[candidate_path]
        deltas = [candidate[image_id] - baseline[image_id] for image_id in image_ids]
        summary = bootstrap_delta(deltas, args.resamples, args.seed + offset)
        summary.update({
            "baseline_path": baseline_path,
            "candidate_path": candidate_path,
            "baseline_dataset_coco_ap": metric_cache[baseline_path]["AP"],
            "candidate_dataset_coco_ap": metric_cache[candidate_path]["AP"],
            "dataset_coco_ap_delta": (
                metric_cache[candidate_path]["AP"] - metric_cache[baseline_path]["AP"]
            ),
        })
        report["comparisons"][name] = summary
        row = {"comparison": name}
        row.update(summary)
        rows.append(row)
        print(name, summary, flush=True)

    os.makedirs(args.output_dir, exist_ok=True)
    atomic_dump(report, os.path.join(args.output_dir, "paired_bootstrap_report.json"), indent=2)
    csv_path = os.path.join(args.output_dir, "paired_bootstrap_summary.csv")
    temporary = csv_path + ".tmp"
    with open(temporary, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, csv_path)


if __name__ == "__main__":
    main()
