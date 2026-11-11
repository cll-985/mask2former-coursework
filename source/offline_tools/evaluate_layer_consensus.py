#!/usr/bin/env python3
"""Evaluate instance consensus across decoder layers in one Mask2Former pass.

The script captures the predictor's existing ``aux_outputs`` and final output,
post-processes selected decoder layers independently, and applies the same
fixed instance-consensus rule used by ``fuse_coco_instances.py``.  It does not
change model weights and reuses one backbone/decoder forward pass per image.

Layer indices follow ``diagnose_decoder_layers.py``: auxiliary outputs first,
then the final output.  For the standard 9-layer decoder this yields 0..9.
"""

from __future__ import print_function

import argparse
import json
import os
import sys
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch
from pycocotools import mask as mask_util
from torch.nn import functional as F

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, build_detection_test_loader
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.structures import ImageList

from mask2former import add_maskformer2_config
from offline_tools.fuse_coco_instances import evaluate_results, fuse_results
from train_cardd import CARDD_SPLITS, register_cardd
from train_net import Trainer


def parse_layers(value):
    layers = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        layer = int(token)
        if layer < 0:
            raise argparse.ArgumentTypeError("layer indices must be non-negative")
        if layer not in layers:
            layers.append(layer)
    if len(layers) < 2:
        raise argparse.ArgumentTypeError("at least two distinct layers are required")
    return layers


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--dataset", default="cardd_val")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--layers", type=parse_layers, default=parse_layers("3,9"))
    parser.add_argument("--match-iou", type=float, default=0.5)
    parser.add_argument("--max-dets", type=int, default=100)
    parser.add_argument("--max-images", type=int, default=0,
                        help="0 evaluates the full registered split")
    parser.add_argument("--seed", type=int, default=42)
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
    cfg.SEED = args.seed
    cfg.freeze()
    return cfg


def xyxy_to_xywh(box):
    x0, y0, x1, y1 = [float(value) for value in box]
    return [x0, y0, x1 - x0, y1 - y0]


def instances_to_coco(instances, image_id, contiguous_to_dataset):
    instances = instances.to(torch.device("cpu"))
    boxes = instances.pred_boxes.tensor.numpy()
    scores = instances.scores.tolist()
    classes = instances.pred_classes.tolist()
    masks = instances.pred_masks.numpy().astype(np.uint8)
    results = []
    for box, score, category, binary_mask in zip(boxes, scores, classes, masks):
        rle = mask_util.encode(np.asfortranarray(binary_mask))
        if isinstance(rle["counts"], bytes):
            rle["counts"] = rle["counts"].decode("utf-8")
        results.append({
            "image_id": int(image_id),
            "category_id": int(contiguous_to_dataset.get(int(category), int(category))),
            "bbox": xyxy_to_xywh(box),
            "score": float(score),
            "segmentation": rle,
        })
    return results


def atomic_json_dump(value, path, indent=None):
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    temporary = path + ".tmp"
    with open(temporary, "w") as handle:
        json.dump(value, handle, indent=indent, sort_keys=indent is not None)
    os.replace(temporary, path)


def main():
    args = parse_args()
    if not 0.0 <= args.match_iou <= 1.0:
        raise ValueError("match-iou must be in [0, 1]")
    if args.max_dets <= 0 or args.max_images < 0:
        raise ValueError("max-dets must be positive and max-images non-negative")

    register_cardd()
    cfg = build_cfg(args)
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model = Trainer.build_model(cfg)
    DetectionCheckpointer(model).load(args.weights)
    model.eval()

    # Building the COCO loader materializes the dataset-id to contiguous-id
    # mapping on MetadataCatalog.  Read it only after loader construction.
    loader = build_detection_test_loader(cfg, args.dataset)
    metadata = MetadataCatalog.get(args.dataset)
    dataset_to_contiguous = getattr(
        metadata, "thing_dataset_id_to_contiguous_id", {})
    contiguous_to_dataset = {
        int(contiguous): int(dataset)
        for dataset, contiguous in dataset_to_contiguous.items()
    }

    captured = {}

    def capture_outputs(module, inputs, output):
        del module, inputs
        captured["output"] = output

    hook = model.sem_seg_head.predictor.register_forward_hook(capture_outputs)
    per_layer = {layer: [] for layer in args.layers}
    processed_images = 0
    start_time = time.time()

    try:
        with torch.no_grad():
            for batch in loader:
                if args.max_images and processed_images >= args.max_images:
                    break
                if len(batch) != 1:
                    raise RuntimeError("Layer consensus requires test batch size 1")

                captured.clear()
                model(batch)
                outputs = captured.get("output")
                if outputs is None:
                    raise RuntimeError("Predictor hook did not capture outputs")
                layers = list(outputs.get("aux_outputs", [])) + [{
                    "pred_logits": outputs["pred_logits"],
                    "pred_masks": outputs["pred_masks"],
                }]
                missing = [layer for layer in args.layers if layer >= len(layers)]
                if missing:
                    raise ValueError(
                        "requested layers {} but model exposes 0..{}".format(
                            missing, len(layers) - 1))

                item = batch[0]
                image_id = int(item["image_id"])
                normalized = (item["image"].to(model.device) - model.pixel_mean) / model.pixel_std
                images = ImageList.from_tensors([normalized], model.size_divisibility)
                padded_size = images.tensor.shape[-2:]
                image_size = images.image_sizes[0]
                height = int(item.get("height", image_size[0]))
                width = int(item.get("width", image_size[1]))

                for layer_index in args.layers:
                    layer = layers[layer_index]
                    mask_cls = layer["pred_logits"][0]
                    mask_pred = F.interpolate(
                        layer["pred_masks"], size=padded_size,
                        mode="bilinear", align_corners=False)[0]
                    mask_pred = sem_seg_postprocess(
                        mask_pred, image_size, height, width)
                    mask_cls = mask_cls.to(mask_pred)
                    instances = model.instance_inference(mask_cls, mask_pred)
                    per_layer[layer_index].extend(instances_to_coco(
                        instances, image_id, contiguous_to_dataset))

                processed_images += 1
                if processed_images % 25 == 0:
                    print("processed_images={}".format(processed_images), flush=True)
    finally:
        hook.remove()

    elapsed = time.time() - start_time
    layer_metrics = {}
    for layer_index in args.layers:
        path = os.path.join(
            args.output_dir, "layer{}_coco_instances_results.json".format(layer_index))
        atomic_json_dump(per_layer[layer_index], path)
        if args.max_images == 0:
            layer_metrics[str(layer_index)] = evaluate_results(
                CARDD_SPLITS[args.dataset][0], path)

    fused = per_layer[args.layers[0]]
    pair_reports = []
    for layer_index in args.layers[1:]:
        fused, pair_report = fuse_results(
            fused, per_layer[layer_index], args.match_iou, args.max_dets)
        pair_report["added_layer"] = layer_index
        pair_reports.append(pair_report)

    fused_path = os.path.join(args.output_dir, "fused_coco_instances_results.json")
    atomic_json_dump(fused, fused_path)
    report = {
        "dataset": args.dataset,
        "weights": os.path.abspath(args.weights),
        "seed": args.seed,
        "layers": args.layers,
        "match_iou": args.match_iou,
        "max_dets": args.max_dets,
        "max_images": args.max_images,
        "processed_images": processed_images,
        "elapsed_seconds": elapsed,
        "detections_per_layer": {
            str(layer): len(per_layer[layer]) for layer in args.layers
        },
        "pair_reports": pair_reports,
        "layer_evaluation": layer_metrics,
    }
    if args.max_images == 0:
        report["fused_evaluation"] = evaluate_results(
            CARDD_SPLITS[args.dataset][0], fused_path)
    atomic_json_dump(
        report, os.path.join(args.output_dir, "layer_consensus_report.json"), indent=2)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
