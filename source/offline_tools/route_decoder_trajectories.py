#!/usr/bin/env python3
"""Evaluate fixed, final-anchored routing rules for decoder trajectories.

The early and final layer detections are aligned by category and mask IoU.  All
rules preserve the final layer as the anchor, so unmatched early detections do
not globally displace mature predictions.  The three fixed rules test distinct
hypotheses without retraining:

* final_anchor_max: choose the higher-score endpoint for each matched track;
* stability_weighted: additionally weight matched scores by inter-layer IoU;
* small_recovery: apply consensus only when both endpoint masks are COCO-small,
  while leaving medium/large final detections unchanged.
"""

from __future__ import print_function

import argparse
import copy
import json
import os
import sys
from collections import Counter, defaultdict

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from pycocotools import mask as mask_util

from offline_tools.fuse_coco_instances import evaluate_results, greedy_matches


RULES = ("final_anchor_max", "stability_weighted", "small_recovery")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--early-results", required=True)
    parser.add_argument("--final-results", required=True)
    parser.add_argument("--gt-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--match-iou", type=float, default=0.5)
    parser.add_argument("--small-area", type=float, default=32.0 * 32.0)
    parser.add_argument("--max-dets", type=int, default=100)
    return parser.parse_args()


def grouped(results):
    output = defaultdict(list)
    for item in results:
        output[(int(item["image_id"]), int(item["category_id"]))].append(item)
    return output


def mask_area(item):
    rle = dict(item["segmentation"])
    if isinstance(rle.get("counts"), str):
        rle["counts"] = rle["counts"].encode("utf-8")
    return float(mask_util.area(rle))


def prune_topk(items, max_dets):
    by_image = defaultdict(list)
    for item in items:
        by_image[int(item["image_id"])].append(item)
    output = []
    for image_id in sorted(by_image):
        ranked = sorted(
            by_image[image_id],
            key=lambda x: (-float(x["score"]), int(x["category_id"])))
        output.extend(ranked[:max_dets])
    return output


def route(early_results, final_results, rule, match_iou, small_area, max_dets):
    early_groups = grouped(early_results)
    final_groups = grouped(final_results)
    output = []
    stats = Counter()

    for key in sorted(set(early_groups) | set(final_groups)):
        early = sorted(early_groups[key], key=lambda x: -float(x["score"]))
        final = sorted(final_groups[key], key=lambda x: -float(x["score"]))
        matches = greedy_matches(early, final, match_iou)
        matched_early = set()
        matched_final = set()

        for early_index, final_index, track_iou in matches:
            a = early[early_index]
            b = final[final_index]
            matched_early.add(early_index)
            matched_final.add(final_index)
            stats["matched"] += 1
            stats["matched_iou_sum"] += float(track_iou)

            if rule == "small_recovery":
                is_small_track = max(mask_area(a), mask_area(b)) < small_area
                if is_small_track:
                    item = copy.deepcopy(
                        a if float(a["score"]) > float(b["score"]) else b)
                    item["score"] = 0.5 * (float(a["score"]) + float(b["score"]))
                    stats["small_tracks"] += 1
                    stats["early_selected"] += int(item is not None and float(a["score"]) > float(b["score"]))
                else:
                    item = copy.deepcopy(b)
            else:
                choose_early = float(a["score"]) > float(b["score"])
                item = copy.deepcopy(a if choose_early else b)
                stats["early_selected"] += int(choose_early)
                if rule == "stability_weighted":
                    # Map IoU 0.5..1.0 to a reliability factor 0.75..1.0.
                    item["score"] = float(item["score"]) * (0.5 + 0.5 * float(track_iou))
            output.append(item)

        # The final layer is always the anchor and is never discarded merely
        # because an early-layer counterpart is absent.
        for index, item in enumerate(final):
            if index not in matched_final:
                output.append(copy.deepcopy(item))
                stats["unmatched_final_kept"] += 1

        # Only the scale-gated rule may recover an unmatched early prediction,
        # and only when its predicted mask is COCO-small.
        if rule == "small_recovery":
            for index, item in enumerate(early):
                if index in matched_early or mask_area(item) >= small_area:
                    continue
                recovered = copy.deepcopy(item)
                recovered["score"] = 0.5 * float(recovered["score"])
                output.append(recovered)
                stats["unmatched_early_small_added"] += 1

    report = {
        "rule": rule,
        "matched": int(stats["matched"]),
        "mean_matched_iou": float(stats["matched_iou_sum"]) / max(int(stats["matched"]), 1),
        "early_selected": int(stats["early_selected"]),
        "small_tracks": int(stats["small_tracks"]),
        "unmatched_final_kept": int(stats["unmatched_final_kept"]),
        "unmatched_early_small_added": int(stats["unmatched_early_small_added"]),
    }
    return prune_topk(output, max_dets), report


def atomic_dump(value, path, indent=None):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temporary = path + ".tmp"
    with open(temporary, "w") as handle:
        json.dump(value, handle, indent=indent, sort_keys=indent is not None)
    os.replace(temporary, path)


def main():
    args = parse_args()
    if not 0.0 <= args.match_iou <= 1.0:
        raise ValueError("match-iou must be in [0, 1]")
    if args.small_area <= 0 or args.max_dets <= 0:
        raise ValueError("small-area and max-dets must be positive")
    with open(args.early_results, "r") as handle:
        early = json.load(handle)
    with open(args.final_results, "r") as handle:
        final = json.load(handle)

    summary = {}
    for rule in RULES:
        results, report = route(
            early, final, rule, args.match_iou, args.small_area, args.max_dets)
        result_path = os.path.join(args.output_dir, rule + "_coco_instances_results.json")
        atomic_dump(results, result_path)
        report["output_detections"] = len(results)
        report["evaluation"] = evaluate_results(args.gt_json, result_path)
        summary[rule] = report
        print(rule, report["evaluation"]["segm"], flush=True)
    atomic_dump(
        summary, os.path.join(args.output_dir, "trajectory_routing_report.json"), indent=2)


if __name__ == "__main__":
    main()
