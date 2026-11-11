#!/usr/bin/env python3
"""Train a conservative trajectory router on train and freeze it for val.

No validation ground truth is used for fitting, feature normalization, label
construction, or threshold selection.  The final decoder output is the anchor;
the router changes a matched detection to its early-layer endpoint only when a
train-calibrated logistic model has sufficient confidence.
"""

from __future__ import print_function

import argparse
import copy
import hashlib
import json
import os
import sys
from collections import Counter

import numpy as np
from pycocotools.coco import COCO

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from offline_tools.fuse_coco_instances import evaluate_results, greedy_matches
from offline_tools.validate_learned_trajectory_router import (
    atomic_dump,
    feature_vector,
    grouped,
    build_tracks,
    evaluate_subset,
    sigmoid,
)


THRESHOLDS = (0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 1.00)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-early-results", required=True)
    parser.add_argument("--train-final-results", required=True)
    parser.add_argument("--train-gt-json", required=True)
    parser.add_argument("--val-early-results", required=True)
    parser.add_argument("--val-final-results", required=True)
    parser.add_argument("--val-gt-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split-seed", type=int, default=20261007)
    parser.add_argument("--tune-fraction", type=float, default=0.20)
    parser.add_argument("--match-iou", type=float, default=0.50)
    parser.add_argument("--label-margin", type=float, default=0.05)
    parser.add_argument("--quality-floor", type=float, default=0.50)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--l2", type=float, default=0.01)
    return parser.parse_args()


def stable_train_tune_split(image_ids, seed, tune_fraction):
    def key(image_id):
        token = (str(seed) + ":" + str(int(image_id))).encode("utf-8")
        return hashlib.sha256(token).hexdigest()

    ordered = sorted(image_ids, key=key)
    tune_count = max(1, int(round(len(ordered) * tune_fraction)))
    return set(ordered[tune_count:]), set(ordered[:tune_count])


def build_unlabelled_tracks(early_results, final_results, category_ids, match_iou):
    early_groups = grouped(early_results)
    final_groups = grouped(final_results)
    tracks = []
    unmatched_final = []
    for key in sorted(set(early_groups) | set(final_groups)):
        early = sorted(early_groups[key], key=lambda item: -float(item["score"]))
        final = sorted(final_groups[key], key=lambda item: -float(item["score"]))
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
            })
        for index, item in enumerate(final):
            if index not in matched_final:
                unmatched_final.append(item)
    return tracks, unmatched_final


def fit_conservative_router(tracks, fit_image_ids, label_margin, quality_floor,
                            epochs, learning_rate, l2):
    selected = [track for track in tracks if track["image_id"] in fit_image_ids]
    if not selected:
        raise RuntimeError("No fit tracks")
    x = np.asarray([track["features"] for track in selected], dtype=np.float64)
    y = np.asarray([
        track["early_quality"] >= quality_floor and
        track["early_quality"] - track["final_quality"] >= label_margin
        for track in selected
    ], dtype=np.float64)
    positives = int(np.sum(y))
    negatives = int(len(y) - positives)
    if positives == 0:
        raise RuntimeError("No positive conservative routing labels")

    continuous_count = 7
    mean = np.mean(x[:, :continuous_count], axis=0)
    scale = np.std(x[:, :continuous_count], axis=0)
    scale[scale < 1e-8] = 1.0
    x[:, :continuous_count] = (x[:, :continuous_count] - mean) / scale
    x = np.concatenate([np.ones((x.shape[0], 1)), x], axis=1)

    # A square-root positive weight exposes the rare useful transitions while
    # avoiding the severe over-selection caused by fully balanced weighting.
    positive_weight = float(np.sqrt(float(negatives) / float(positives)))
    sample_weights = np.where(y > 0.5, positive_weight, 1.0)
    weights = np.zeros(x.shape[1], dtype=np.float64)
    for _ in range(epochs):
        probabilities = sigmoid(np.dot(x, weights))
        gradient = np.dot(x.T, sample_weights * (probabilities - y)) / len(y)
        regularizer = l2 * weights
        regularizer[0] = 0.0
        weights -= learning_rate * (gradient + regularizer)

    return {
        "weights": weights,
        "mean": mean,
        "scale": scale,
        "continuous_count": continuous_count,
        "examples": len(selected),
        "positives": positives,
        "negatives": negatives,
        "positive_rate": float(positives) / float(len(selected)),
        "positive_weight": positive_weight,
    }


def predict_probability(track, model):
    values = np.asarray(track["features"], dtype=np.float64)
    count = model["continuous_count"]
    values[:count] = (values[:count] - model["mean"]) / model["scale"]
    values = np.concatenate([[1.0], values])
    return float(sigmoid(np.dot(values, model["weights"])))


def route(tracks, unmatched_final, image_ids, model, threshold):
    output = []
    stats = Counter()
    for track in tracks:
        if track["image_id"] not in image_ids:
            continue
        probability = predict_probability(track, model)
        use_early = probability >= threshold
        item = copy.deepcopy(track["early"] if use_early else track["final"])
        # Keep final-layer confidence so routing changes geometry only and does
        # not silently recalibrate the global COCO ranking.
        item["score"] = float(track["final"]["score"])
        output.append(item)
        stats["matched"] += 1
        stats["early_selected"] += int(use_early)
    for item in unmatched_final:
        if int(item["image_id"]) in image_ids:
            output.append(copy.deepcopy(item))
            stats["unmatched_final_kept"] += 1
    return output, stats


def serializable_model(model, category_ids, threshold):
    return {
        "weights": model["weights"].tolist(),
        "feature_mean": model["mean"].tolist(),
        "feature_scale": model["scale"].tolist(),
        "continuous_feature_count": model["continuous_count"],
        "category_ids": list(category_ids),
        "threshold": float(threshold),
        "examples": model["examples"],
        "positives": model["positives"],
        "negatives": model["negatives"],
        "positive_rate": model["positive_rate"],
        "positive_weight": model["positive_weight"],
    }


def main():
    args = parse_args()
    if not 0.0 < args.tune_fraction < 1.0:
        raise ValueError("tune-fraction must be in (0, 1)")
    with open(args.train_early_results, "r") as handle:
        train_early = json.load(handle)
    with open(args.train_final_results, "r") as handle:
        train_final = json.load(handle)
    with open(args.val_early_results, "r") as handle:
        val_early = json.load(handle)
    with open(args.val_final_results, "r") as handle:
        val_final = json.load(handle)

    train_coco = COCO(args.train_gt_json)
    val_coco = COCO(args.val_gt_json)
    train_ids = sorted(set(int(item["image_id"]) for item in train_final))
    fit_ids, tune_ids = stable_train_tune_split(
        train_ids, args.split_seed, args.tune_fraction)
    train_tracks, train_unmatched, category_ids = build_tracks(
        train_early, train_final, train_coco, args.match_iou)
    model = fit_conservative_router(
        train_tracks, fit_ids, args.label_margin, args.quality_floor,
        args.epochs, args.learning_rate, args.l2)

    tune_results = {}
    for threshold in THRESHOLDS:
        predictions, stats = route(
            train_tracks, train_unmatched, tune_ids, model, threshold)
        metrics = evaluate_subset(train_coco, predictions, tune_ids)["segm"]
        tune_results[str(threshold)] = {
            "metrics": metrics,
            "early_selected": int(stats["early_selected"]),
            "matched": int(stats["matched"]),
        }
    selected_threshold = max(
        THRESHOLDS,
        key=lambda value: (
            tune_results[str(value)]["metrics"]["AP"],
            value,
        ),
    )

    val_tracks, val_unmatched = build_unlabelled_tracks(
        val_early, val_final, category_ids, args.match_iou)
    val_ids = set(int(item["image_id"]) for item in val_final)
    routed, route_stats = route(
        val_tracks, val_unmatched, val_ids, model, selected_threshold)
    routed_path = os.path.join(args.output_dir, "val_routed_coco_instances_results.json")
    atomic_dump(routed, routed_path)
    model_path = os.path.join(args.output_dir, "frozen_router.json")
    atomic_dump(
        serializable_model(model, category_ids, selected_threshold),
        model_path,
        indent=2,
    )

    val_final_metrics = evaluate_results(args.val_gt_json, args.val_final_results)
    val_early_metrics = evaluate_results(args.val_gt_json, args.val_early_results)
    val_routed_metrics = evaluate_results(args.val_gt_json, routed_path)
    report = {
        "protocol": {
            "split_seed": args.split_seed,
            "fit_images": len(fit_ids),
            "tune_images": len(tune_ids),
            "val_images": len(val_ids),
            "match_iou": args.match_iou,
            "label_margin": args.label_margin,
            "quality_floor": args.quality_floor,
            "threshold_candidates": list(THRESHOLDS),
            "selected_threshold": selected_threshold,
            "validation_ground_truth_used_for_training": False,
        },
        "model": serializable_model(model, category_ids, selected_threshold),
        "tune_results": tune_results,
        "val": {
            "early": val_early_metrics,
            "final": val_final_metrics,
            "routed": val_routed_metrics,
            "route_stats": {key: int(value) for key, value in route_stats.items()},
        },
    }
    atomic_dump(
        report,
        os.path.join(args.output_dir, "train_only_router_report.json"),
        indent=2,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
