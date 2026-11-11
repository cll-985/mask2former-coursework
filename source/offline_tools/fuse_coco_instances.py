#!/usr/bin/env python3
"""Fuse two COCO instance-result files after class and mask-IoU matching.

The fixed consensus rule is intentionally simple and auditable:

* greedy one-to-one matching within each image and category at mask IoU >= 0.5;
* matched score = arithmetic mean, mask/bbox = the higher-score endpoint;
* unmatched score = original score / 2 (one vote out of two models);
* keep the top 100 fused detections per image.

This operates on post-processed instances and never assumes aligned query IDs.
"""

from __future__ import print_function

import argparse
import copy
import json
import os
from collections import Counter, defaultdict

import numpy as np
from pycocotools import mask as mask_util
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-results", required=True)
    parser.add_argument("--other-results", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--gt-json", required=True)
    parser.add_argument("--match-iou", type=float, default=0.5)
    parser.add_argument("--max-dets", type=int, default=100)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def rle_for_compute(segmentation):
    rle = dict(segmentation)
    if isinstance(rle.get("counts"), str):
        rle["counts"] = rle["counts"].encode("utf-8")
    return rle


def greedy_matches(base_items, other_items, threshold):
    if not base_items or not other_items:
        return []
    base_rles = [rle_for_compute(x["segmentation"]) for x in base_items]
    other_rles = [rle_for_compute(x["segmentation"]) for x in other_items]
    ious = mask_util.iou(base_rles, other_rles, [0] * len(other_rles))
    candidates = []
    for base_index in range(ious.shape[0]):
        for other_index in range(ious.shape[1]):
            value = float(ious[base_index, other_index])
            if value >= threshold:
                candidates.append((-value, base_index, other_index))
    candidates.sort()
    used_base = set()
    used_other = set()
    matches = []
    for negative_iou, base_index, other_index in candidates:
        if base_index in used_base or other_index in used_other:
            continue
        used_base.add(base_index)
        used_other.add(other_index)
        matches.append((base_index, other_index, -negative_iou))
    return matches


def fuse_group(base_items, other_items, threshold, stats):
    matches = greedy_matches(base_items, other_items, threshold)
    matched_base = set()
    matched_other = set()
    fused = []

    for base_index, other_index, mask_iou in matches:
        base_item = base_items[base_index]
        other_item = other_items[other_index]
        matched_base.add(base_index)
        matched_other.add(other_index)
        if float(base_item["score"]) >= float(other_item["score"]):
            item = copy.deepcopy(base_item)
            stats["matched_mask_from_base"] += 1
        else:
            item = copy.deepcopy(other_item)
            stats["matched_mask_from_other"] += 1
        item["score"] = 0.5 * (
            float(base_item["score"]) + float(other_item["score"])
        )
        fused.append(item)
        stats["matched"] += 1
        stats["matched_iou_sum"] += mask_iou

    for index, source in enumerate(base_items):
        if index not in matched_base:
            item = copy.deepcopy(source)
            item["score"] = 0.5 * float(item["score"])
            fused.append(item)
            stats["unmatched_base"] += 1
    for index, source in enumerate(other_items):
        if index not in matched_other:
            item = copy.deepcopy(source)
            item["score"] = 0.5 * float(item["score"])
            fused.append(item)
            stats["unmatched_other"] += 1
    return fused


def fuse_results(base_results, other_results, match_iou, max_dets):
    grouped_base = defaultdict(list)
    grouped_other = defaultdict(list)
    image_ids = set()
    for item in base_results:
        key = (int(item["image_id"]), int(item["category_id"]))
        grouped_base[key].append(item)
        image_ids.add(key[0])
    for item in other_results:
        key = (int(item["image_id"]), int(item["category_id"]))
        grouped_other[key].append(item)
        image_ids.add(key[0])

    stats = Counter()
    per_category = defaultdict(Counter)
    fused_by_image = defaultdict(list)
    all_keys = sorted(set(grouped_base) | set(grouped_other))
    for key in all_keys:
        image_id, category_id = key
        local_stats = Counter()
        fused = fuse_group(
            sorted(grouped_base[key], key=lambda x: -float(x["score"])),
            sorted(grouped_other[key], key=lambda x: -float(x["score"])),
            match_iou,
            local_stats,
        )
        stats.update(local_stats)
        per_category[category_id].update(local_stats)
        fused_by_image[image_id].extend(fused)

    output = []
    before_prune = 0
    for image_id in sorted(image_ids):
        items = fused_by_image[image_id]
        before_prune += len(items)
        items.sort(
            key=lambda x: (
                -float(x["score"]),
                int(x["category_id"]),
                str(x["segmentation"].get("counts", "")),
            )
        )
        output.extend(items[:max_dets])

    report = {
        "input_base_detections": len(base_results),
        "input_other_detections": len(other_results),
        "fused_before_topk": before_prune,
        "fused_after_topk": len(output),
        "images": len(image_ids),
        "match_iou": match_iou,
        "max_dets_per_image": max_dets,
        "matched": int(stats["matched"]),
        "mean_matched_iou": (
            float(stats["matched_iou_sum"]) / max(int(stats["matched"]), 1)
        ),
        "matched_mask_from_base": int(stats["matched_mask_from_base"]),
        "matched_mask_from_other": int(stats["matched_mask_from_other"]),
        "unmatched_base": int(stats["unmatched_base"]),
        "unmatched_other": int(stats["unmatched_other"]),
        "per_category": {
            str(category_id): {key: float(value) for key, value in values.items()}
            for category_id, values in sorted(per_category.items())
        },
    }
    return output, report


def evaluate_results(gt_json, results_path):
    coco_gt = COCO(gt_json)
    with open(results_path, "r") as handle:
        predictions = json.load(handle)
    summary = {}
    for task in ("bbox", "segm"):
        # Match Detectron2's COCOEvaluator semantics.  COCO.loadRes assigns a
        # detection's area from bbox whenever bbox is present.  For segm this
        # would silently use box area for APs/APm/APl, even though overall AP is
        # unchanged.  Detectron2 removes bbox before loading segmentation
        # results so that mask area defines the size bins.
        task_predictions = predictions
        if task == "segm":
            task_predictions = []
            for prediction in predictions:
                item = copy.deepcopy(prediction)
                item.pop("bbox", None)
                task_predictions.append(item)
        coco_dt = coco_gt.loadRes(task_predictions)
        evaluator = COCOeval(coco_gt, coco_dt, task)
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
        values = [float(x) for x in evaluator.stats]
        task_summary = {
            "AP": 100.0 * values[0],
            "AP50": 100.0 * values[1],
            "AP75": 100.0 * values[2],
            "APs": 100.0 * values[3],
            "APm": 100.0 * values[4],
            "APl": 100.0 * values[5],
        }
        precision = evaluator.eval["precision"]
        category_ids = evaluator.params.catIds
        per_category = {}
        for category_index, category_id in enumerate(category_ids):
            category_precision = precision[:, :, category_index, 0, -1]
            valid = category_precision[category_precision > -1]
            per_category[str(category_id)] = (
                100.0 * float(np.mean(valid)) if valid.size else float("nan")
            )
        task_summary["per_category_AP"] = per_category
        summary[task] = task_summary
    return summary


def encode_mask(mask):
    rle = mask_util.encode(np.asfortranarray(mask.astype(np.uint8)))
    if isinstance(rle["counts"], bytes):
        rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def self_test():
    a = np.zeros((8, 8), dtype=np.uint8)
    a[1:4, 1:4] = 1
    b = a.copy()
    c = np.zeros((8, 8), dtype=np.uint8)
    c[5:7, 5:7] = 1
    base = [
        {"image_id": 1, "category_id": 1, "bbox": [1, 1, 3, 3],
         "score": 0.8, "segmentation": encode_mask(a)}
    ]
    other = [
        {"image_id": 1, "category_id": 1, "bbox": [1, 1, 3, 3],
         "score": 0.6, "segmentation": encode_mask(b)},
        {"image_id": 1, "category_id": 1, "bbox": [5, 5, 2, 2],
         "score": 0.4, "segmentation": encode_mask(c)},
    ]
    fused, report = fuse_results(base, other, 0.5, 100)
    assert len(fused) == 2
    assert abs(fused[0]["score"] - 0.7) < 1e-8
    assert abs(fused[1]["score"] - 0.2) < 1e-8
    assert report["matched"] == 1
    assert report["unmatched_other"] == 1
    print("FUSION_SELF_TEST_PASS")


def main():
    args = parse_args()
    if args.self_test:
        self_test()
        return
    if not 0.0 <= args.match_iou <= 1.0:
        raise ValueError("match-iou must be in [0, 1]")
    if args.max_dets <= 0:
        raise ValueError("max-dets must be positive")

    with open(args.base_results, "r") as handle:
        base_results = json.load(handle)
    with open(args.other_results, "r") as handle:
        other_results = json.load(handle)
    fused_results, report = fuse_results(
        base_results, other_results, args.match_iou, args.max_dets
    )

    output_path = os.path.abspath(args.output)
    report_path = os.path.abspath(args.report)
    for path in (output_path, report_path):
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
    temporary_output = output_path + ".tmp"
    with open(temporary_output, "w") as handle:
        json.dump(fused_results, handle)
    os.replace(temporary_output, output_path)

    report["evaluation"] = evaluate_results(args.gt_json, output_path)
    temporary_report = report_path + ".tmp"
    with open(temporary_report, "w") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    os.replace(temporary_report, report_path)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
