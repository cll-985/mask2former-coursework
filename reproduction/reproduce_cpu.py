#!/usr/bin/env python3
"""Regenerate frozen fusion and evaluate saved CarDD test predictions on CPU.

No images, torch, CUDA, or Detectron2 are needed. Only the two original
prediction JSONs and the original annotation JSON are required. This script
reuses the audited original fusion/evaluator functions without changing them.
"""
from __future__ import print_function

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import types

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parent
if (PACKAGE / ".deps").is_dir():
    sys.path.insert(0, str(PACKAGE / ".deps"))
# Original compare_fusion_rules imports offline_tools.fuse_coco_instances.
# Bind the preserved local module under that name instead of importing the
# entire research project or Detectron2.
import fuse_coco_instances as fusion
namespace = types.ModuleType("offline_tools")
namespace.__path__ = [str(HERE)]
sys.modules.setdefault("offline_tools", namespace)
sys.modules["offline_tools.fuse_coco_instances"] = fusion
spec = importlib.util.spec_from_file_location(
    "frozen_original_rules", str(HERE / "compare_fusion_rules_original.py"))
rules = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rules)

RUN_RELATIVE = "output_offline/Q3_empirical_evidence/M4_frozen_test_20261009_113405"
KEYS = ("AP", "AP50", "AP75", "APs", "APm", "APl")
# Values rounded in the contemporaneous report; tolerance covers rounding only.
EXPECTED = {
    "seed3407": (51.0354, 67.8498, 50.8110, 7.9224, 11.8538, 54.3146),
    "seed42": (50.8553, 68.3505, 50.5796, 13.7490, 10.4512, 54.5798),
    "mask_nms": (52.7653, 70.6358, 53.0174, 14.7997, 12.4524, 56.5738),
    "consensus": (52.8984, 70.6319, 53.2657, 8.5177, 13.0749, 57.0056),
}


def sha256(path):
    digest = hashlib.sha256()
    with open(str(path), "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    with open(str(path), "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(value, path):
    with open(str(path), "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", type=Path, default=PACKAGE / "evidence")
    parser.add_argument("--annotation", type=Path,
                        help="official frozen CarDD test annotation JSON obtained separately")
    parser.add_argument("--output-dir", type=Path, default=PACKAGE / "verification" / "cpu_reproduction")
    parser.add_argument("--tolerance", type=float, default=0.0005,
                        help="maximum AP-point deviation from rounded historical metrics")
    parser.add_argument("--keep-predictions", action="store_true",
                        help="keep regenerated JSONs (otherwise remove after verification to save disk)")
    args = parser.parse_args()
    if args.tolerance < 0 or args.tolerance > 0.001:
        parser.error("tolerance must be between 0 and 0.001 AP points")
    run = args.evidence_root / RUN_RELATIVE
    gt = args.annotation or args.evidence_root / "CarDD_COCO/annotations/instances_test2017.json"
    protocol_path = run / "protocol.json"
    inputs = {
        "seed3407": run / "seed3407/inference/coco_instances_results.json",
        "seed42": run / "seed42/inference/coco_instances_results.json",
    }
    for path in [gt, protocol_path] + list(inputs.values()):
        if not path.is_file():
            raise FileNotFoundError(str(path))
    # Never overwrite archived observations with a new evaluation.
    out = args.output_dir.resolve()
    try:
        out.relative_to(args.evidence_root.resolve())
    except ValueError:
        pass
    else:
        raise ValueError("output-dir must be outside the immutable evidence directory")
    out.mkdir(parents=True, exist_ok=True)
    protocol = read_json(protocol_path)
    gt_hash = sha256(gt)
    if gt_hash.lower() != protocol["test_annotation_sha256"].lower():
        raise ValueError("Test annotation SHA256 differs from frozen protocol")
    if protocol["mask_iou_threshold"] != 0.5 or protocol["max_detections_per_image"] != 100:
        raise ValueError("Archived protocol differs from the frozen 0.5/top100 rule")
    gt_data = read_json(gt)
    if len(gt_data["images"]) != 374 or len(gt_data["annotations"]) != 785:
        raise ValueError("Unexpected test split size")
    report = {
        "test_annotation_sha256": gt_hash,
        "source_protocol_sha256": sha256(protocol_path),
        "input_prediction_sha256": {name: sha256(path) for name, path in inputs.items()},
        "parameters": {"mask_iou_threshold": 0.5, "max_detections_per_image": 100},
        "segmentation_size_semantics": "remove bbox before COCO.loadRes; detection area uses mask",
        "metrics": {}, "canonical_prediction_matches": {}, "metric_errors_ap_points": {},
    }
    manifest_path = run / "release-input-sha256.json"
    if manifest_path.is_file():
        manifest = read_json(manifest_path)
        actual = dict(report["input_prediction_sha256"], protocol=sha256(protocol_path))
        if actual != manifest:
            raise ValueError("Archived prediction or protocol SHA256 differs from the release manifest")
    base = read_json(inputs["seed3407"])
    other = read_json(inputs["seed42"])
    generated_nms, nms_report = rules.mask_nms(base, other, 0.5, 100)
    generated_consensus, consensus_report = fusion.fuse_results(base, other, 0.5, 100)
    report["fusion_statistics"] = {"mask_nms": nms_report, "consensus": consensus_report}
    for rule, generated in (("mask_nms", generated_nms), ("consensus", generated_consensus)):
        output = out / (rule + "_coco_instances_results.json")
        write_json(generated, output)
        inputs[rule] = output
        archived = run / "fusion" / (rule + "_coco_instances_results.json")
        if archived.is_file():
            # Compare decoded JSON, because whitespace/key order are immaterial.
            report["canonical_prediction_matches"][rule] = generated == read_json(archived)
    failed = []
    for name, path in inputs.items():
        print("Evaluating", name, flush=True)
        metrics = fusion.evaluate_results(str(gt), str(path))
        report["metrics"][name] = metrics
        errors = {key: metrics["segm"][key] - value
                  for key, value in zip(KEYS, EXPECTED[name])}
        report["metric_errors_ap_points"][name] = errors
        if any(abs(value) > args.tolerance for value in errors.values()):
            failed.append(name)
        write_json(report, out / "cpu_reproduction_report.json")
        if name in ("mask_nms", "consensus") and not args.keep_predictions:
            path.unlink()
    mismatch = [name for name, value in report["canonical_prediction_matches"].items() if not value]
    report["delta_mask_nms_vs_seed3407_ap"] = (
        report["metrics"]["mask_nms"]["segm"]["AP"] -
        report["metrics"]["seed3407"]["segm"]["AP"])
    report["passed"] = not failed and not mismatch
    report["failure_reasons"] = {"metric_mismatches": failed, "prediction_mismatches": mismatch}
    write_json(report, out / "cpu_reproduction_report.json")
    print("CPU_REPRODUCTION_" + ("PASS" if report["passed"] else "FAIL"), flush=True)
    print("Report:", out / "cpu_reproduction_report.json")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
