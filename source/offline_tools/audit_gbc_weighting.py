#!/usr/bin/env python3
"""Audit GBC geometry weights after the real CarDD LSJ data pipeline.

This is a read-only diagnostic: it builds the existing training loader, samples
augmented ground-truth masks, resizes them to the mask-feature resolution, and
reports the exact geometry weights used by GBC-v1.
"""

from __future__ import print_function

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.env import seed_all_rng

from mask2former import add_maskformer2_config
from mask2former.modeling.criterion import geometry_weights
from train_cardd import register_cardd
from train_net import Trainer


CLASS_NAMES = [
    "dent",
    "scratch",
    "crack",
    "glass shatter",
    "lamp broken",
    "tire flat",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-file",
        default="configs/cardd/maskformer2_R50_cardd_gbc_v1.yaml",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 3407])
    parser.add_argument("--images-per-seed", type=int, default=800)
    parser.add_argument("--mask-resolution", type=int, default=256)
    parser.add_argument(
        "--output",
        default="audit/cardd/gbc_v1_weight_audit.json",
    )
    return parser.parse_args()


def percentile_dict(values):
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"n": 0}
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "min": float(array.min()),
        "p10": float(np.percentile(array, 10)),
        "p25": float(np.percentile(array, 25)),
        "median": float(np.percentile(array, 50)),
        "p75": float(np.percentile(array, 75)),
        "p90": float(np.percentile(array, 90)),
        "max": float(array.max()),
    }


def audit_seed(cfg, seed, images_limit, mask_resolution):
    seed_all_rng(seed)
    cfg = cfg.clone()
    cfg.defrost()
    cfg.SEED = seed
    # A single-process mapper makes the audit deterministic and avoids worker
    # prefetch changing how many random transforms are consumed.
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.freeze()

    loader = Trainer.build_train_loader(cfg)
    all_weights = []
    all_area_ratios = []
    batch_weight_means = []
    by_class = {name: [] for name in CLASS_NAMES}
    by_ratio = {"lt_0.001": [], "0.001_to_0.01": [], "0.01_to_0.1": [], "ge_0.1": []}
    images_seen = 0

    for batch in loader:
        batch_masks = []
        batch_classes = []
        for sample in batch:
            if images_seen >= images_limit:
                break
            images_seen += 1
            instances = sample["instances"]
            gt_masks = instances.gt_masks
            if hasattr(gt_masks, "tensor"):
                gt_masks = gt_masks.tensor
            masks = gt_masks[:, None].float()
            if masks.numel() == 0:
                continue
            batch_masks.append(masks)
            batch_classes.append(instances.gt_classes.cpu())

        if batch_masks:
            masks = F.interpolate(
                torch.cat(batch_masks, dim=0),
                size=(mask_resolution, mask_resolution),
                mode="nearest",
            )
            weights = geometry_weights(
                masks,
                small_area_ratio=cfg.MODEL.MASK_FORMER.GBC.SMALL_AREA_RATIO,
                min_weight=cfg.MODEL.MASK_FORMER.GBC.GEO_WEIGHT_MIN,
                max_weight=cfg.MODEL.MASK_FORMER.GBC.GEO_WEIGHT_MAX,
                normalize=cfg.MODEL.MASK_FORMER.GBC.NORMALIZE_WEIGHTS,
            ).cpu()
            areas = (masks > 0.5).flatten(1).sum(1).cpu()
            ratios = areas / float(mask_resolution * mask_resolution)
            classes = torch.cat(batch_classes, dim=0)

            for weight, ratio, class_id in zip(weights, ratios, classes):
                weight_value = float(weight.item())
                ratio_value = float(ratio.item())
                all_weights.append(weight_value)
                all_area_ratios.append(ratio_value)
                class_index = int(class_id.item())
                if 0 <= class_index < len(CLASS_NAMES):
                    by_class[CLASS_NAMES[class_index]].append(weight_value)
                if ratio_value < 0.001:
                    ratio_key = "lt_0.001"
                elif ratio_value < 0.01:
                    ratio_key = "0.001_to_0.01"
                elif ratio_value < 0.1:
                    ratio_key = "0.01_to_0.1"
                else:
                    ratio_key = "ge_0.1"
                by_ratio[ratio_key].append(weight_value)

            batch_weight_means.append(float(weights.mean().item()))
        if images_seen >= images_limit:
            break

    return {
        "seed": seed,
        "images": images_seen,
        "instances": len(all_weights),
        "weights": percentile_dict(all_weights),
        "area_ratios": percentile_dict(all_area_ratios),
        "batch_weight_means": percentile_dict(batch_weight_means),
        "by_class": {key: percentile_dict(value) for key, value in by_class.items()},
        "by_area_ratio": {key: percentile_dict(value) for key, value in by_ratio.items()},
    }


def main():
    args = parse_args()
    register_cardd()
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.freeze()

    result = {
        "config_file": args.config_file,
        "images_per_seed": args.images_per_seed,
        "mask_resolution": args.mask_resolution,
        "small_area_ratio": cfg.MODEL.MASK_FORMER.GBC.SMALL_AREA_RATIO,
        "geo_weight_min": cfg.MODEL.MASK_FORMER.GBC.GEO_WEIGHT_MIN,
        "geo_weight_max": cfg.MODEL.MASK_FORMER.GBC.GEO_WEIGHT_MAX,
        "normalize_weights": cfg.MODEL.MASK_FORMER.GBC.NORMALIZE_WEIGHTS,
        "seeds": [
            audit_seed(cfg, seed, args.images_per_seed, args.mask_resolution)
            for seed in args.seeds
        ],
    }
    output_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    print(json.dumps(result, indent=2, sort_keys=True))
    print("Wrote {}".format(output_path))


if __name__ == "__main__":
    main()
