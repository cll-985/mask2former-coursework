#!/usr/bin/env python3
"""Benchmark final-only, single-forward cross-layer, and two-model inference.

The benchmark caches a fixed validation subset in host memory, warms up CUDA,
and synchronizes around every measured image.  Cross-layer timing includes the
extra layer post-processing and the fixed instance-consensus rule.  The
two-model latency is the sum of independently measured single-model latencies,
which represents sequential deployment without GPU contention.
"""

from __future__ import print_function

import argparse
import json
import os
import statistics
import sys
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, build_detection_test_loader
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.structures import ImageList
from torch.nn import functional as F

from mask2former import add_maskformer2_config
from offline_tools.evaluate_layer_consensus import instances_to_coco
from offline_tools.fuse_coco_instances import fuse_results
from train_cardd import register_cardd
from train_net import Trainer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--weights-primary", required=True)
    parser.add_argument("--weights-secondary", required=True)
    parser.add_argument("--dataset", default="cardd_val")
    parser.add_argument("--output", required=True)
    parser.add_argument("--layer", type=int, default=3)
    parser.add_argument("--match-iou", type=float, default=0.5)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--max-images", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20261009)
    parser.add_argument("opts", nargs=argparse.REMAINDER)
    return parser.parse_args()


def build_cfg(args, weights):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.defrost()
    cfg.MODEL.WEIGHTS = weights
    cfg.DATASETS.TEST = (args.dataset,)
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.SEED = args.seed
    cfg.freeze()
    return cfg


def percentile(values, q):
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def summarize(times_ms, peak_bytes, model):
    params = sum(parameter.numel() for parameter in model.parameters())
    return {
        "images": len(times_ms),
        "latency_ms_mean": float(statistics.mean(times_ms)),
        "latency_ms_std": float(statistics.pstdev(times_ms)),
        "latency_ms_p50": percentile(times_ms, 50),
        "latency_ms_p95": percentile(times_ms, 95),
        "throughput_images_per_second": 1000.0 / float(statistics.mean(times_ms)),
        "peak_cuda_allocated_mib": float(peak_bytes) / (1024.0 ** 2),
        "parameters": int(params),
        "parameter_mib_fp32": float(params * 4) / (1024.0 ** 2),
    }


def load_model(cfg, weights):
    model = Trainer.build_model(cfg)
    DetectionCheckpointer(model).load(weights)
    model.eval()
    return model


def time_final(model, batches, warmup):
    with torch.no_grad():
        for batch in batches[:warmup]:
            model(batch)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    timings = []
    with torch.no_grad():
        for batch in batches[warmup:]:
            torch.cuda.synchronize()
            start = time.perf_counter()
            model(batch)
            torch.cuda.synchronize()
            timings.append((time.perf_counter() - start) * 1000.0)
    return summarize(timings, torch.cuda.max_memory_allocated(), model)


def time_cross_layer(model, batches, warmup, layer_index, match_iou,
                     contiguous_to_dataset):
    captured = {}

    def capture_outputs(module, inputs, output):
        del module, inputs
        captured["output"] = output

    hook = model.sem_seg_head.predictor.register_forward_hook(capture_outputs)

    def run_one(batch):
        captured.clear()
        final_output = model(batch)
        raw = captured.get("output")
        if raw is None:
            raise RuntimeError("predictor hook did not capture decoder output")
        layers = list(raw.get("aux_outputs", [])) + [{
            "pred_logits": raw["pred_logits"],
            "pred_masks": raw["pred_masks"],
        }]
        if layer_index >= len(layers):
            raise ValueError("layer {} unavailable; valid range is 0..{}".format(
                layer_index, len(layers) - 1))
        item = batch[0]
        normalized = ((item["image"].to(model.device) - model.pixel_mean)
                      / model.pixel_std)
        images = ImageList.from_tensors([normalized], model.size_divisibility)
        padded_size = images.tensor.shape[-2:]
        image_size = images.image_sizes[0]
        height = int(item.get("height", image_size[0]))
        width = int(item.get("width", image_size[1]))
        selected = layers[layer_index]
        mask_cls = selected["pred_logits"][0]
        mask_pred = F.interpolate(
            selected["pred_masks"], size=padded_size,
            mode="bilinear", align_corners=False)[0]
        mask_pred = sem_seg_postprocess(mask_pred, image_size, height, width)
        layer_instances = model.instance_inference(mask_cls.to(mask_pred), mask_pred)
        image_id = int(item["image_id"])
        final_instances = final_output[0]["instances"]
        final_coco = instances_to_coco(
            final_instances, image_id, contiguous_to_dataset)
        layer_coco = instances_to_coco(
            layer_instances, image_id, contiguous_to_dataset)
        fuse_results(final_coco, layer_coco, match_iou, 100)

    try:
        with torch.no_grad():
            for batch in batches[:warmup]:
                run_one(batch)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        timings = []
        with torch.no_grad():
            for batch in batches[warmup:]:
                torch.cuda.synchronize()
                start = time.perf_counter()
                run_one(batch)
                torch.cuda.synchronize()
                timings.append((time.perf_counter() - start) * 1000.0)
        return summarize(timings, torch.cuda.max_memory_allocated(), model)
    finally:
        hook.remove()


def main():
    args = parse_args()
    if args.warmup < 1 or args.max_images < 1:
        raise ValueError("warmup and max-images must be positive")
    if not 0.0 <= args.match_iou <= 1.0:
        raise ValueError("match-iou must be in [0, 1]")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    register_cardd()

    primary_cfg = build_cfg(args, args.weights_primary)
    loader = build_detection_test_loader(primary_cfg, args.dataset)
    batches = []
    needed = args.warmup + args.max_images
    for batch in loader:
        if len(batch) != 1:
            raise RuntimeError("benchmark requires test batch size 1")
        batches.append(batch)
        if len(batches) >= needed:
            break
    if len(batches) <= args.warmup:
        raise RuntimeError("dataset is too small for the requested warmup")

    metadata = MetadataCatalog.get(args.dataset)
    mapping = getattr(metadata, "thing_dataset_id_to_contiguous_id", {})
    contiguous_to_dataset = {
        int(contiguous): int(dataset) for dataset, contiguous in mapping.items()
    }

    primary = load_model(primary_cfg, args.weights_primary)
    primary_final = time_final(primary, batches, args.warmup)
    primary_cross = time_cross_layer(
        primary, batches, args.warmup, args.layer, args.match_iou,
        contiguous_to_dataset)
    del primary
    torch.cuda.empty_cache()

    secondary_cfg = build_cfg(args, args.weights_secondary)
    secondary = load_model(secondary_cfg, args.weights_secondary)
    secondary_final = time_final(secondary, batches, args.warmup)
    del secondary
    torch.cuda.empty_cache()

    dual_mean = (primary_final["latency_ms_mean"]
                 + secondary_final["latency_ms_mean"])
    report = {
        "protocol": {
            "dataset": args.dataset,
            "seed": args.seed,
            "warmup_images": args.warmup,
            "measured_images": len(batches) - args.warmup,
            "cuda_device_name": torch.cuda.get_device_name(0),
            "torch_version": torch.__version__,
            "config_file": os.path.abspath(args.config_file),
            "primary_weights": os.path.abspath(args.weights_primary),
            "secondary_weights": os.path.abspath(args.weights_secondary),
            "cross_layer": args.layer,
            "match_iou": args.match_iou,
        },
        "final_only_primary": primary_final,
        "single_forward_cross_layer": primary_cross,
        "final_only_secondary": secondary_final,
        "derived_two_model_sequential": {
            "latency_ms_mean": dual_mean,
            "throughput_images_per_second": 1000.0 / dual_mean,
            "resident_parameter_mib_fp32": (
                primary_final["parameter_mib_fp32"]
                + secondary_final["parameter_mib_fp32"]),
            "peak_cuda_allocated_mib_sequential": max(
                primary_final["peak_cuda_allocated_mib"],
                secondary_final["peak_cuda_allocated_mib"]),
        },
    }
    report["comparisons"] = {
        "cross_layer_latency_overhead_percent": 100.0 * (
            primary_cross["latency_ms_mean"]
            / primary_final["latency_ms_mean"] - 1.0),
        "cross_layer_peak_memory_overhead_percent": 100.0 * (
            primary_cross["peak_cuda_allocated_mib"]
            / primary_final["peak_cuda_allocated_mib"] - 1.0),
        "cross_layer_speedup_vs_two_model": (
            dual_mean / primary_cross["latency_ms_mean"]),
    }
    output_dir = os.path.dirname(os.path.abspath(args.output))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    temporary = args.output + ".tmp"
    with open(temporary, "w") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    os.replace(temporary, args.output)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
