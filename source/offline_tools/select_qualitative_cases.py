#!/usr/bin/env python3
"""Select and render deterministic qualitative success/failure cases.

Cases are ranked over the complete split by candidate-minus-baseline per-image
mask AP.  This avoids hand-picking visually attractive examples after looking
at predictions.  The script writes the full ranking, the selected cases, and
three GT/baseline/candidate montages.
"""

from __future__ import print_function

import argparse
import csv
import json
import os
import sys
from collections import defaultdict

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pycocotools import mask as mask_util
from pycocotools.coco import COCO

from offline_tools.fuse_coco_instances import rle_for_compute
from offline_tools.paired_image_bootstrap_mask_ap import (
    image_scores,
    load_ground_truth,
    load_predictions,
)


COLORS = [
    (231, 76, 60), (52, 152, 219), (46, 204, 113),
    (241, 196, 15), (155, 89, 182), (230, 126, 34),
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-json", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--score-threshold", type=float, default=0.30)
    parser.add_argument("--max-predictions", type=int, default=15)
    return parser.parse_args()


def decode_prediction(item):
    mask = mask_util.decode(rle_for_compute(item["segmentation"]))
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    return mask.astype(bool)


def blend_masks(image, masks):
    output = np.asarray(image.convert("RGB"), dtype=np.float32).copy()
    for mask, category_id in masks:
        color = np.asarray(COLORS[(int(category_id) - 1) % len(COLORS)],
                           dtype=np.float32)
        output[mask] = 0.58 * output[mask] + 0.42 * color
    return Image.fromarray(np.clip(output, 0, 255).astype(np.uint8))


def render_gt(coco, image, image_id):
    masks = []
    labels = []
    for annotation in coco.loadAnns(coco.getAnnIds(imgIds=[image_id])):
        if int(annotation.get("iscrowd", 0)):
            continue
        masks.append((coco.annToMask(annotation).astype(bool),
                      int(annotation["category_id"])))
        labels.append((int(annotation["category_id"]), 1.0))
    return blend_masks(image, masks), labels


def render_predictions(image, image_id, grouped, threshold, maximum):
    items = []
    for (candidate_image_id, _), values in grouped.items():
        if candidate_image_id == image_id:
            items.extend(values)
    items = sorted(items, key=lambda value: -float(value["score"]))
    items = [value for value in items if float(value["score"]) >= threshold][:maximum]
    masks = [(decode_prediction(item), int(item["category_id"])) for item in items]
    labels = [(int(item["category_id"]), float(item["score"])) for item in items]
    return blend_masks(image, masks), labels


def add_title(image, title, subtitle):
    canvas = Image.new("RGB", (image.width, image.height + 48), "white")
    canvas.paste(image, (0, 48))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((8, 6), title, fill="black", font=font)
    draw.text((8, 25), subtitle, fill=(60, 60, 60), font=font)
    return canvas


def fit_panel(image, width=520, height=390):
    image = image.copy()
    image.thumbnail((width, height), Image.BILINEAR)
    canvas = Image.new("RGB", (width, height), (245, 245, 245))
    x = (width - image.width) // 2
    y = (height - image.height) // 2
    canvas.paste(image, (x, y))
    return canvas


def make_montage(coco, image_root, records, baseline_grouped,
                 candidate_grouped, args, path, label):
    rows = []
    image_metadata = {int(item["id"]): item for item in coco.dataset["images"]}
    for record in records:
        image_id = int(record["image_id"])
        metadata = image_metadata[image_id]
        source = Image.open(os.path.join(image_root, metadata["file_name"])).convert("RGB")
        gt, gt_labels = render_gt(coco, source, image_id)
        baseline, baseline_labels = render_predictions(
            source, image_id, baseline_grouped,
            args.score_threshold, args.max_predictions)
        candidate, candidate_labels = render_predictions(
            source, image_id, candidate_grouped,
            args.score_threshold, args.max_predictions)
        common = "id={} {} delta={:+.2f}".format(
            image_id, metadata["file_name"], record["delta"])
        panels = [
            add_title(fit_panel(gt), "Ground truth", common),
            add_title(fit_panel(baseline), "Baseline",
                      "image AP={:.2f}; shown={}".format(
                          record["baseline_image_ap"], len(baseline_labels))),
            add_title(fit_panel(candidate), "Output-space mask-NMS",
                      "image AP={:.2f}; shown={}".format(
                          record["candidate_image_ap"], len(candidate_labels))),
        ]
        row = Image.new("RGB", (sum(panel.width for panel in panels),
                                max(panel.height for panel in panels)), "white")
        x = 0
        for panel in panels:
            row.paste(panel, (x, 0))
            x += panel.width
        rows.append(row)
    montage = Image.new("RGB", (max(row.width for row in rows),
                                sum(row.height for row in rows) + 36), "white")
    draw = ImageDraw.Draw(montage)
    draw.text((8, 10), label, fill="black", font=ImageFont.load_default())
    y = 36
    for row in rows:
        montage.paste(row, (0, y))
        y += row.height
    montage.save(path)


def main():
    args = parse_args()
    if args.count <= 0:
        raise ValueError("count must be positive")
    coco = COCO(args.gt_json)
    ground_truth = load_ground_truth(coco)
    baseline_grouped = load_predictions(args.baseline)
    candidate_grouped = load_predictions(args.candidate)
    baseline_scores = image_scores(coco, ground_truth, baseline_grouped)
    candidate_scores = image_scores(coco, ground_truth, candidate_grouped)
    metadata = {int(item["id"]): item for item in coco.dataset["images"]}
    gt_counts = defaultdict(int)
    for annotation in coco.dataset["annotations"]:
        if not int(annotation.get("iscrowd", 0)):
            gt_counts[int(annotation["image_id"])] += 1

    records = []
    for image_id in sorted(int(value) for value in coco.getImgIds()):
        if gt_counts[image_id] <= 0:
            continue
        baseline = float(baseline_scores[image_id])
        candidate = float(candidate_scores[image_id])
        records.append({
            "image_id": image_id,
            "file_name": metadata[image_id]["file_name"],
            "ground_truth_instances": gt_counts[image_id],
            "baseline_image_ap": baseline,
            "candidate_image_ap": candidate,
            "delta": candidate - baseline,
        })
    ranked = sorted(records, key=lambda value: (-value["delta"], value["image_id"]))
    for rank, record in enumerate(ranked, 1):
        record["rank_descending"] = rank
    successes = ranked[:args.count]
    failures = sorted(records, key=lambda value: (value["delta"], value["image_id"]))[:args.count]
    selected_ids = {record["image_id"] for record in successes + failures}
    neutral_pool = [record for record in records if record["image_id"] not in selected_ids]
    neutrals = sorted(neutral_pool,
                      key=lambda value: (abs(value["delta"]), value["image_id"]))[:args.count]

    os.makedirs(args.output_dir, exist_ok=True)
    fieldnames = [
        "rank_descending", "image_id", "file_name", "ground_truth_instances",
        "baseline_image_ap", "candidate_image_ap", "delta",
    ]
    with open(os.path.join(args.output_dir, "full_image_delta_ranking.csv"),
              "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(ranked)
    selection = {
        "protocol": {
            "selection": "top/bottom/closest-to-zero candidate-minus-baseline per-image mask AP",
            "count_per_group": args.count,
            "score_threshold_for_rendering": args.score_threshold,
            "max_predictions_for_rendering": args.max_predictions,
            "baseline": os.path.abspath(args.baseline),
            "candidate": os.path.abspath(args.candidate),
        },
        "success": successes,
        "failure": failures,
        "neutral": neutrals,
    }
    with open(os.path.join(args.output_dir, "selected_cases.json"), "w") as handle:
        json.dump(selection, handle, indent=2, sort_keys=True)
    make_montage(coco, args.image_root, successes, baseline_grouped,
                 candidate_grouped, args,
                 os.path.join(args.output_dir, "success_cases.png"),
                 "Deterministic top improvements")
    make_montage(coco, args.image_root, failures, baseline_grouped,
                 candidate_grouped, args,
                 os.path.join(args.output_dir, "failure_cases.png"),
                 "Deterministic largest regressions")
    make_montage(coco, args.image_root, neutrals, baseline_grouped,
                 candidate_grouped, args,
                 os.path.join(args.output_dir, "neutral_cases.png"),
                 "Deterministic near-zero changes")
    print(json.dumps(selection, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
