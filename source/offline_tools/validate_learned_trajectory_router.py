#!/usr/bin/env python3
"""Validate a learned decoder-trajectory router without test-set leakage.

The script makes a deterministic image-level split.  A logistic router is
trained only on the calibration-train images, its decision threshold is chosen
only on calibration-tune images, and the final AP delta is reported only on
held-out images.  At inference time the router uses prediction-side features
only; ground truth is used solely to create calibration labels and metrics.

The final decoder layer remains the anchor.  For each class-aware matched
layer-3/layer-9 track, the router either keeps the final endpoint or replaces
it with the early endpoint while retaining the final confidence score.
Unmatched final detections are always preserved and unmatched early detections
are discarded.
"""

from __future__ import print_function

import argparse
import copy
import hashlib
import json
import math
import os
import sys
from collections import Counter, defaultdict

import numpy as np
from pycocotools import mask as mask_util
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from offline_tools.fuse_coco_instances import greedy_matches, rle_for_compute


THRESHOLDS = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--early-results", required=True)
    parser.add_argument("--final-results", required=True)
    parser.add_argument("--gt-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split-seed", type=int, default=20261007)
    parser.add_argument("--heldout-fraction", type=float, default=0.5)
    parser.add_argument("--tune-fraction-of-calibration", type=float, default=0.2)
    parser.add_argument("--match-iou", type=float, default=0.5)
    parser.add_argument("--label-margin", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--l2", type=float, default=0.01)
    return parser.parse_args()


def atomic_dump(value, path, indent=None):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temporary = path + ".tmp"
    with open(temporary, "w") as handle:
        json.dump(value, handle, indent=indent, sort_keys=indent is not None)
    os.replace(temporary, path)


def grouped(results):
    output = defaultdict(list)
    for item in results:
        output[(int(item["image_id"]), int(item["category_id"]))].append(item)
    return output


def mask_area(item):
    return float(mask_util.area(rle_for_compute(item["segmentation"])))


def stable_split(image_ids, seed, heldout_fraction, tune_fraction):
    def key(image_id):
        token = (str(seed) + ":" + str(int(image_id))).encode("utf-8")
        return hashlib.sha256(token).hexdigest()

    ordered = sorted(image_ids, key=key)
    heldout_count = int(round(len(ordered) * heldout_fraction))
    heldout = set(ordered[:heldout_count])
    calibration = ordered[heldout_count:]
    tune_count = max(1, int(round(len(calibration) * tune_fraction)))
    tune = set(calibration[:tune_count])
    train = set(calibration[tune_count:])
    return train, tune, heldout


def gt_rles(coco, image_id, category_id, cache):
    key = (int(image_id), int(category_id))
    if key not in cache:
        annotation_ids = coco.getAnnIds(imgIds=[key[0]], catIds=[key[1]])
        annotations = coco.loadAnns(annotation_ids)
        cache[key] = (
            [coco.annToRLE(annotation) for annotation in annotations],
            [int(annotation.get("iscrowd", 0)) for annotation in annotations],
        )
    return cache[key]


def endpoint_quality(item, coco, cache):
    rles, crowds = gt_rles(coco, item["image_id"], item["category_id"], cache)
    if not rles:
        return 0.0
    values = mask_util.iou([rle_for_compute(item["segmentation"])], rles, crowds)
    return float(np.max(values)) if values.size else 0.0


def feature_vector(early, final, track_iou, category_ids):
    early_score = float(early["score"])
    final_score = float(final["score"])
    early_area = max(mask_area(early), 1.0)
    final_area = max(mask_area(final), 1.0)
    continuous = [
        early_score,
        final_score,
        early_score - final_score,
        float(track_iou),
        math.log(early_area),
        math.log(final_area),
        math.log(early_area / final_area),
    ]
    one_hot = [1.0 if int(early["category_id"]) == category_id else 0.0
               for category_id in category_ids]
    return continuous + one_hot


def build_tracks(early_results, final_results, coco, match_iou):
    early_groups = grouped(early_results)
    final_groups = grouped(final_results)
    category_ids = sorted(int(value) for value in coco.getCatIds())
    cache = {}
    tracks = []
    unmatched_final = []
    for key in sorted(set(early_groups) | set(final_groups)):
        early = sorted(early_groups[key], key=lambda x: -float(x["score"]))
        final = sorted(final_groups[key], key=lambda x: -float(x["score"]))
        matches = greedy_matches(early, final, match_iou)
        matched_final = set()
        for early_index, final_index, track_iou in matches:
            a = early[early_index]
            b = final[final_index]
            matched_final.add(final_index)
            tracks.append({
                "image_id": int(key[0]),
                "early": a,
                "final": b,
                "track_iou": float(track_iou),
                "features": feature_vector(a, b, track_iou, category_ids),
                "early_quality": endpoint_quality(a, coco, cache),
                "final_quality": endpoint_quality(b, coco, cache),
            })
        for index, item in enumerate(final):
            if index not in matched_final:
                unmatched_final.append(item)
    return tracks, unmatched_final, category_ids


def sigmoid(values):
    values = np.clip(values, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-values))


def fit_router(tracks, image_ids, label_margin, epochs, learning_rate, l2):
    selected = [track for track in tracks if track["image_id"] in image_ids]
    labelled = [track for track in selected if
                abs(track["early_quality"] - track["final_quality"]) > label_margin]
    if not labelled:
        raise RuntimeError("No non-tied calibration tracks")
    x = np.asarray([track["features"] for track in labelled], dtype=np.float64)
    y = np.asarray([track["early_quality"] > track["final_quality"]
                    for track in labelled], dtype=np.float64)
    continuous_count = 7
    mean = np.mean(x[:, :continuous_count], axis=0)
    scale = np.std(x[:, :continuous_count], axis=0)
    scale[scale < 1e-8] = 1.0
    x[:, :continuous_count] = (x[:, :continuous_count] - mean) / scale
    x = np.concatenate([np.ones((x.shape[0], 1)), x], axis=1)

    positive = max(float(np.sum(y)), 1.0)
    negative = max(float(len(y) - np.sum(y)), 1.0)
    sample_weights = np.where(y > 0.5, 0.5 * len(y) / positive,
                              0.5 * len(y) / negative)
    weights = np.zeros(x.shape[1], dtype=np.float64)
    for _ in range(epochs):
        probabilities = sigmoid(np.dot(x, weights))
        gradient = np.dot(x.T, sample_weights * (probabilities - y)) / len(y)
        regularizer = l2 * weights
        regularizer[0] = 0.0
        weights -= learning_rate * (gradient + regularizer)
    predictions = sigmoid(np.dot(x, weights)) >= 0.5
    accuracy = float(np.mean(predictions == (y > 0.5)))
    return {
        "weights": weights,
        "mean": mean,
        "scale": scale,
        "continuous_count": continuous_count,
        "examples": len(labelled),
        "positive_rate": float(np.mean(y)),
        "balanced_training_accuracy": accuracy,
    }


def predict_probability(track, model):
    values = np.asarray(track["features"], dtype=np.float64)
    count = model["continuous_count"]
    values[:count] = (values[:count] - model["mean"]) / model["scale"]
    values = np.concatenate([[1.0], values])
    return float(sigmoid(np.dot(values, model["weights"])))


def quality_for_threshold(tracks, image_ids, model, threshold):
    selected = [track for track in tracks if track["image_id"] in image_ids]
    if not selected:
        return float("nan")
    qualities = []
    for track in selected:
        use_early = predict_probability(track, model) >= threshold
        qualities.append(track["early_quality"] if use_early else track["final_quality"])
    return float(np.mean(qualities))


def route_tracks(tracks, unmatched_final, image_ids, model, threshold, oracle=False):
    output = []
    stats = Counter()
    for track in tracks:
        if track["image_id"] not in image_ids:
            continue
        if oracle:
            use_early = track["early_quality"] > track["final_quality"]
        else:
            use_early = predict_probability(track, model) >= threshold
        item = copy.deepcopy(track["early"] if use_early else track["final"])
        # Preserve the mature decoder's ranking confidence; the router selects
        # geometry, not an independently recalibrated confidence score.
        item["score"] = float(track["final"]["score"])
        output.append(item)
        stats["matched"] += 1
        stats["early_selected"] += int(use_early)
    for item in unmatched_final:
        if int(item["image_id"]) in image_ids:
            output.append(copy.deepcopy(item))
            stats["unmatched_final_kept"] += 1
    return output, stats


def evaluate_subset(coco, predictions, image_ids):
    image_ids = sorted(int(value) for value in image_ids)
    filtered = [item for item in predictions if int(item["image_id"]) in set(image_ids)]
    summary = {}
    for task in ("bbox", "segm"):
        task_predictions = []
        for prediction in filtered:
            item = copy.deepcopy(prediction)
            if task == "segm":
                item.pop("bbox", None)
            task_predictions.append(item)
        coco_dt = coco.loadRes(task_predictions)
        evaluator = COCOeval(coco, coco_dt, task)
        evaluator.params.imgIds = image_ids
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
        values = [float(value) for value in evaluator.stats]
        summary[task] = {
            "AP": 100.0 * values[0],
            "AP50": 100.0 * values[1],
            "AP75": 100.0 * values[2],
            "APs": 100.0 * values[3],
            "APm": 100.0 * values[4],
            "APl": 100.0 * values[5],
        }
    return summary


def serializable_model(model):
    return {
        "weights": model["weights"].tolist(),
        "feature_mean": model["mean"].tolist(),
        "feature_scale": model["scale"].tolist(),
        "examples": model["examples"],
        "positive_rate": model["positive_rate"],
        "balanced_training_accuracy": model["balanced_training_accuracy"],
    }


def main():
    args = parse_args()
    if not 0.0 < args.heldout_fraction < 1.0:
        raise ValueError("heldout-fraction must be in (0, 1)")
    if not 0.0 < args.tune_fraction_of_calibration < 1.0:
        raise ValueError("tune-fraction-of-calibration must be in (0, 1)")
    with open(args.early_results, "r") as handle:
        early_results = json.load(handle)
    with open(args.final_results, "r") as handle:
        final_results = json.load(handle)
    coco = COCO(args.gt_json)
    image_ids = sorted(set(int(item["image_id"]) for item in final_results))
    train_ids, tune_ids, heldout_ids = stable_split(
        image_ids, args.split_seed, args.heldout_fraction,
        args.tune_fraction_of_calibration)
    tracks, unmatched_final, category_ids = build_tracks(
        early_results, final_results, coco, args.match_iou)
    model = fit_router(
        tracks, train_ids, args.label_margin, args.epochs,
        args.learning_rate, args.l2)

    tune_scores = {str(threshold): quality_for_threshold(
        tracks, tune_ids, model, threshold) for threshold in THRESHOLDS}
    threshold = max(THRESHOLDS, key=lambda value: (tune_scores[str(value)], value))

    routed, routed_stats = route_tracks(
        tracks, unmatched_final, heldout_ids, model, threshold)
    oracle, oracle_stats = route_tracks(
        tracks, unmatched_final, heldout_ids, model, threshold, oracle=True)
    early_heldout = [item for item in early_results
                     if int(item["image_id"]) in heldout_ids]
    final_heldout = [item for item in final_results
                     if int(item["image_id"]) in heldout_ids]

    output_path = os.path.join(args.output_dir, "heldout_routed_results.json")
    atomic_dump(routed, output_path)
    report = {
        "protocol": {
            "split_seed": args.split_seed,
            "train_images": len(train_ids),
            "tune_images": len(tune_ids),
            "heldout_images": len(heldout_ids),
            "match_iou": args.match_iou,
            "label_margin": args.label_margin,
            "threshold_candidates": list(THRESHOLDS),
            "selected_threshold": threshold,
            "category_ids": category_ids,
        },
        "model": serializable_model(model),
        "tune_mean_endpoint_iou": tune_scores,
        "heldout": {
            "early": evaluate_subset(coco, early_heldout, heldout_ids),
            "final": evaluate_subset(coco, final_heldout, heldout_ids),
            "routed": evaluate_subset(coco, routed, heldout_ids),
            "oracle_endpoint": evaluate_subset(coco, oracle, heldout_ids),
            "routed_stats": {key: int(value) for key, value in routed_stats.items()},
            "oracle_stats": {key: int(value) for key, value in oracle_stats.items()},
        },
    }
    atomic_dump(report, os.path.join(args.output_dir, "heldout_router_report.json"), indent=2)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
