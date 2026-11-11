#!/usr/bin/env python3
"""Export layer-wise Mask2Former query/GT diagnostics without changing the model.

The script attaches a forward hook to the transformer predictor and reads the
existing aux_outputs. It never changes weights and only evaluates cardd_val.
"""

from __future__ import print_function

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict

# When invoked as ``python offline_tools/diagnose_decoder_layers.py``, Python
# puts offline_tools/ (not the repository root) on sys.path. Add the root
# explicitly before importing the local mask2former/train_cardd modules.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch
from torch.nn import functional as F
from pycocotools import mask as mask_util

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, build_detection_test_loader
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.structures import ImageList

from mask2former import add_maskformer2_config
from train_cardd import register_cardd
from train_net import Trainer


FIELDS = (
    "seed", "image_id", "annotation_id", "category_id", "layer", "area",
    "size_bin", "compactness", "gt_pixels", "class_agnostic_query",
    "class_agnostic_iou", "class_aware_query", "class_aware_iou",
    "class_aware_score",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--seed", required=True)
    parser.add_argument("--dataset", default="cardd_val")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-images", type=int, default=0,
                        help="0 means the full validation set")
    parser.add_argument("--max-side", type=int, default=512)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("opts", nargs=argparse.REMAINDER)
    return parser.parse_args()


def build_cfg(args):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.defrost()
    cfg.MODEL.WEIGHTS = args.weights
    cfg.DATASETS.TEST = (args.dataset,)
    cfg.freeze()
    return cfg


def decode_segmentation(segmentation, height, width):
    if isinstance(segmentation, list):
        rles = mask_util.frPyObjects(segmentation, height, width)
        rle = mask_util.merge(rles)
    elif isinstance(segmentation, dict) and isinstance(segmentation.get("counts"), list):
        rle = mask_util.frPyObjects(segmentation, height, width)
    else:
        rle = segmentation
    decoded = mask_util.decode(rle)
    if decoded.ndim == 3:
        decoded = np.any(decoded, axis=2)
    return decoded.astype(np.uint8)


def perimeter_and_compactness(binary_mask):
    area = float(binary_mask.sum())
    padded = np.pad(binary_mask.astype(np.int16), 1, mode="constant")
    perimeter = float(
        np.abs(np.diff(padded, axis=0)).sum()
        + np.abs(np.diff(padded, axis=1)).sum()
    )
    compactness = 0.0 if perimeter <= 0 else 4.0 * math.pi * area / (perimeter * perimeter)
    return perimeter, compactness


def size_bin(area):
    if area < 32.0 * 32.0:
        return "small"
    if area < 96.0 * 96.0:
        return "medium"
    return "large"


def diagnostic_shape(height, width, max_side):
    scale = min(1.0, float(max_side) / float(max(height, width)))
    return max(1, int(round(height * scale))), max(1, int(round(width * scale)))


def best_iou(mask_logits, gt_mask, candidates):
    if candidates.numel() == 0 or gt_mask.sum().item() == 0:
        return -1, float("nan")
    pred = (mask_logits[candidates] > 0).flatten(1).float()
    gt = gt_mask.flatten().float()
    intersection = torch.matmul(pred, gt)
    union = pred.sum(1) + gt.sum() - intersection
    ious = intersection / union.clamp(min=1.0)
    value, position = torch.max(ious, dim=0)
    return int(candidates[position].item()), float(value.item())


def aggregate(rows):
    grouped = defaultdict(list)
    by_instance = defaultdict(dict)
    for row in rows:
        grouped[(int(row["layer"]), row["size_bin"])].append(row)
        grouped[(int(row["layer"]), "all")].append(row)
        by_instance[(row["image_id"], row["annotation_id"])][int(row["layer"])] = row

    layer_summary = []
    for (layer, group), items in sorted(grouped.items()):
        values = [float(x["class_aware_iou"]) for x in items
                  if np.isfinite(float(x["class_aware_iou"]))]
        agnostic = [float(x["class_agnostic_iou"]) for x in items
                    if np.isfinite(float(x["class_agnostic_iou"]))]
        layer_summary.append({
            "layer": layer,
            "group": group,
            "n": len(values),
            "class_aware_mean_iou": float(np.mean(values)) if values else None,
            "class_aware_coverage_50": float(np.mean(np.asarray(values) >= 0.5)) if values else None,
            "class_agnostic_mean_iou": float(np.mean(agnostic)) if agnostic else None,
            "class_agnostic_coverage_50": float(np.mean(np.asarray(agnostic) >= 0.5)) if agnostic else None,
        })

    transitions = defaultdict(lambda: {"n": 0, "stable": 0, "recovered": 0, "degraded": 0})
    for layers in by_instance.values():
        ordered = sorted(layers)
        if len(ordered) < 2:
            continue
        group = layers[ordered[-1]]["size_bin"]
        for label in (group, "all"):
            first = layers[ordered[0]]
            final = layers[ordered[-1]]
            bucket = transitions[label]
            bucket["n"] += 1
            bucket["recovered"] += int(float(first["class_aware_iou"]) < 0.5 <= float(final["class_aware_iou"]))
            bucket["degraded"] += int(float(first["class_aware_iou"]) >= 0.5 > float(final["class_aware_iou"]))
            adjacent = 0
            stable = 0
            for left, right in zip(ordered[:-1], ordered[1:]):
                adjacent += 1
                stable += int(layers[left]["class_aware_query"] == layers[right]["class_aware_query"])
            bucket.setdefault("adjacent", 0)
            bucket["adjacent"] += adjacent
            bucket["stable"] += stable

    transition_summary = {}
    for group, values in transitions.items():
        n = float(values["n"])
        adjacent = float(values.get("adjacent", 0))
        transition_summary[group] = {
            "n": values["n"],
            "recovery_rate": values["recovered"] / n if n else None,
            "degradation_rate": values["degraded"] / n if n else None,
            "adjacent_query_identity_stability": values["stable"] / adjacent if adjacent else None,
        }
    return layer_summary, transition_summary


def main():
    args = parse_args()
    if args.max_side <= 0 or args.topk <= 0:
        raise ValueError("max-side and topk must be positive")
    register_cardd()
    cfg = build_cfg(args)
    os.makedirs(args.output_dir, exist_ok=True)

    model = Trainer.build_model(cfg)
    DetectionCheckpointer(model).load(args.weights)
    model.eval()

    raw_records = {int(x["image_id"]): x for x in DatasetCatalog.get(args.dataset)}
    loader = build_detection_test_loader(cfg, args.dataset)
    captured = {}

    def capture_outputs(module, inputs, output):
        del module, inputs
        captured["output"] = output

    hook = model.sem_seg_head.predictor.register_forward_hook(capture_outputs)
    rows = []
    try:
        with torch.no_grad():
            for image_index, batch in enumerate(loader):
                if args.max_images and image_index >= args.max_images:
                    break
                if len(batch) != 1:
                    raise RuntimeError("Diagnostic requires test batch size 1")
                captured.clear()
                model(batch)
                outputs = captured.get("output")
                if outputs is None:
                    raise RuntimeError("Predictor hook did not capture outputs")

                item = batch[0]
                image_id = int(item["image_id"])
                raw = raw_records[image_id]
                original_h, original_w = int(raw["height"]), int(raw["width"])
                diag_h, diag_w = diagnostic_shape(original_h, original_w, args.max_side)

                normalized = (item["image"].to(model.device) - model.pixel_mean) / model.pixel_std
                images = ImageList.from_tensors([normalized], model.size_divisibility)
                padded_size = images.tensor.shape[-2:]
                transformed_size = images.image_sizes[0]

                gt_items = []
                for annotation_index, annotation in enumerate(raw.get("annotations", [])):
                    gt_np = decode_segmentation(annotation["segmentation"], original_h, original_w)
                    _, compactness = perimeter_and_compactness(gt_np)
                    gt = torch.from_numpy(gt_np[None, None].astype(np.float32)).to(model.device)
                    gt = F.interpolate(gt, size=(diag_h, diag_w), mode="nearest")[0, 0] > 0.5
                    area = float(annotation.get("area", gt_np.sum()))
                    # Detectron2's COCO loader may omit the source annotation id.
                    # The per-image index is stable and prevents multiple instances
                    # from collapsing into the same (image_id, -1) transition key.
                    instance_id = int(annotation.get("id", annotation_index))
                    gt_items.append((annotation, instance_id, gt, area, compactness))

                layers = list(outputs.get("aux_outputs", [])) + [{
                    "pred_logits": outputs["pred_logits"],
                    "pred_masks": outputs["pred_masks"],
                }]
                for layer_index, layer in enumerate(layers):
                    class_prob = layer["pred_logits"][0].softmax(-1)[:, :-1]
                    objectness, _ = class_prob.max(dim=1)
                    agnostic_candidates = torch.topk(
                        objectness, k=min(args.topk, objectness.numel())).indices
                    mask_logits = F.interpolate(
                        layer["pred_masks"], size=padded_size,
                        mode="bilinear", align_corners=False)[0]
                    mask_logits = sem_seg_postprocess(
                        mask_logits, transformed_size, diag_h, diag_w)

                    for annotation, instance_id, gt, area, compactness in gt_items:
                        category_id = int(annotation["category_id"])
                        if category_id >= class_prob.shape[1]:
                            raise RuntimeError("category_id is not contiguous: {}".format(category_id))
                        scores = class_prob[:, category_id]
                        aware_candidates = torch.topk(
                            scores, k=min(args.topk, scores.numel())).indices
                        agnostic_query, agnostic_iou = best_iou(
                            mask_logits, gt, agnostic_candidates)
                        aware_query, aware_iou = best_iou(mask_logits, gt, aware_candidates)
                        aware_score = float(scores[aware_query].item()) if aware_query >= 0 else float("nan")
                        rows.append({
                            "seed": args.seed,
                            "image_id": image_id,
                            "annotation_id": instance_id,
                            "category_id": category_id,
                            "layer": layer_index,
                            "area": area,
                            "size_bin": size_bin(area),
                            "compactness": compactness,
                            "gt_pixels": int(gt.sum().item()),
                            "class_agnostic_query": agnostic_query,
                            "class_agnostic_iou": agnostic_iou,
                            "class_aware_query": aware_query,
                            "class_aware_iou": aware_iou,
                            "class_aware_score": aware_score,
                        })
                if (image_index + 1) % 25 == 0:
                    print("processed_images={}".format(image_index + 1), flush=True)
    finally:
        hook.remove()

    csv_path = os.path.join(args.output_dir, "layerwise_instances_seed{}.csv".format(args.seed))
    with open(csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    layer_summary, transitions = aggregate(rows)
    summary_path = os.path.join(args.output_dir, "layerwise_summary_seed{}.json".format(args.seed))
    with open(summary_path, "w") as handle:
        json.dump({
            "seed": args.seed,
            "weights": os.path.abspath(args.weights),
            "dataset": args.dataset,
            "max_images": args.max_images,
            "max_side": args.max_side,
            "topk": args.topk,
            "rows": len(rows),
            "layer_summary": layer_summary,
            "transitions": transitions,
        }, handle, indent=2, sort_keys=True)
    print("Wrote", csv_path)
    print("Wrote", summary_path)


if __name__ == "__main__":
    main()
