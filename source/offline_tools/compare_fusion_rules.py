#!/usr/bin/env python3
"""Compare fixed, training-free COCO instance fusion baselines.

The rules are deliberately parameter-light and share the same class-aware mask
IoU threshold and per-image top-k budget:

* concat_topk: concatenate detections and retain the top-k scores;
* mask_nms: class-aware greedy mask NMS over concatenated detections;
* matched_no_penalty: merge matched pairs but keep unmatched scores unchanged;
* consensus: merge matched pairs and halve unmatched scores (the P6 rule).

This script is an ablation tool.  It must not be used to tune rules on the
official test split.
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

from offline_tools.fuse_coco_instances import (
    evaluate_results,
    fuse_results,
    greedy_matches,
    rle_for_compute,
)


RULES = ("concat_topk", "mask_nms", "matched_no_penalty", "consensus")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-results", required=True)
    parser.add_argument("--other-results", required=True)
    parser.add_argument("--gt-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--match-iou", type=float, default=0.5)
    parser.add_argument("--max-dets", type=int, default=100)
    parser.add_argument("--rules", default=",".join(RULES))
    return parser.parse_args()


def atomic_json_dump(value, path, indent=None):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temporary = path + ".tmp"
    with open(temporary, "w") as handle:
        json.dump(value, handle, indent=indent, sort_keys=indent is not None)
    os.replace(temporary, path)


def group_results(results):
    grouped = defaultdict(list)
    for item in results:
        grouped[(int(item["image_id"]), int(item["category_id"]))].append(item)
    return grouped


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


def concat_topk(base, other, max_dets):
    return prune_topk([copy.deepcopy(x) for x in base + other], max_dets), {
        "input": len(base) + len(other),
        "rule": "concat_topk",
    }


def mask_nms(base, other, threshold, max_dets):
    grouped = group_results(base + other)
    kept = []
    stats = Counter()
    for key in sorted(grouped):
        candidates = sorted(
            grouped[key], key=lambda x: -float(x["score"]))
        while candidates:
            winner = copy.deepcopy(candidates.pop(0))
            kept.append(winner)
            stats["kept_before_topk"] += 1
            if not candidates:
                continue
            winner_rle = rle_for_compute(winner["segmentation"])
            candidate_rles = [rle_for_compute(x["segmentation"]) for x in candidates]
            ious = mask_util.iou(
                [winner_rle], candidate_rles, [0] * len(candidate_rles))[0]
            survivors = []
            for item, iou in zip(candidates, ious):
                if float(iou) >= threshold:
                    stats["suppressed"] += 1
                else:
                    survivors.append(item)
            candidates = survivors
    return prune_topk(kept, max_dets), {
        "rule": "mask_nms",
        "kept_before_topk": int(stats["kept_before_topk"]),
        "suppressed": int(stats["suppressed"]),
    }


def matched_no_penalty(base, other, threshold, max_dets):
    grouped_base = group_results(base)
    grouped_other = group_results(other)
    output = []
    stats = Counter()
    for key in sorted(set(grouped_base) | set(grouped_other)):
        left = sorted(grouped_base[key], key=lambda x: -float(x["score"]))
        right = sorted(grouped_other[key], key=lambda x: -float(x["score"]))
        matches = greedy_matches(left, right, threshold)
        used_left = set()
        used_right = set()
        for left_index, right_index, mask_iou in matches:
            a = left[left_index]
            b = right[right_index]
            used_left.add(left_index)
            used_right.add(right_index)
            item = copy.deepcopy(a if float(a["score"]) >= float(b["score"]) else b)
            item["score"] = 0.5 * (float(a["score"]) + float(b["score"]))
            output.append(item)
            stats["matched"] += 1
            stats["matched_iou_sum"] += float(mask_iou)
        for index, item in enumerate(left):
            if index not in used_left:
                output.append(copy.deepcopy(item))
                stats["unmatched_base"] += 1
        for index, item in enumerate(right):
            if index not in used_right:
                output.append(copy.deepcopy(item))
                stats["unmatched_other"] += 1
    report = {
        "rule": "matched_no_penalty",
        "matched": int(stats["matched"]),
        "mean_matched_iou": float(stats["matched_iou_sum"]) / max(int(stats["matched"]), 1),
        "unmatched_base": int(stats["unmatched_base"]),
        "unmatched_other": int(stats["unmatched_other"]),
    }
    return prune_topk(output, max_dets), report


def main():
    args = parse_args()
    if not 0.0 <= args.match_iou <= 1.0:
        raise ValueError("match-iou must be in [0, 1]")
    if args.max_dets <= 0:
        raise ValueError("max-dets must be positive")
    requested = [x.strip() for x in args.rules.split(",") if x.strip()]
    unknown = [x for x in requested if x not in RULES]
    if unknown:
        raise ValueError("unknown rules: {}".format(unknown))

    with open(args.base_results, "r") as handle:
        base = json.load(handle)
    with open(args.other_results, "r") as handle:
        other = json.load(handle)
    os.makedirs(args.output_dir, exist_ok=True)
    summary = {}

    for rule in requested:
        if rule == "concat_topk":
            results, report = concat_topk(base, other, args.max_dets)
        elif rule == "mask_nms":
            results, report = mask_nms(
                base, other, args.match_iou, args.max_dets)
        elif rule == "matched_no_penalty":
            results, report = matched_no_penalty(
                base, other, args.match_iou, args.max_dets)
        else:
            results, report = fuse_results(
                base, other, args.match_iou, args.max_dets)
            report["rule"] = "consensus"
        result_path = os.path.join(args.output_dir, rule + "_coco_instances_results.json")
        atomic_json_dump(results, result_path)
        report["output_detections"] = len(results)
        report["evaluation"] = evaluate_results(args.gt_json, result_path)
        summary[rule] = report
        print(rule, report["evaluation"]["segm"], flush=True)

    atomic_json_dump(
        summary, os.path.join(args.output_dir, "fusion_rule_comparison.json"), indent=2)


if __name__ == "__main__":
    main()
