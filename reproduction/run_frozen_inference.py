#!/usr/bin/env python3
"""Re-run both frozen models plus fixed output fusion in a prepared Linux GPU environment.

Run this using the original PyTorch/Detectron2 environment, not the lightweight
CPU metric environment. Paths are parameters; no server or Docker IDs are used.
"""
from __future__ import print_function

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parent
BASELINE = "configs/cardd/maskformer2_R50_cardd_baseline.yaml"
HASHES = {
    "42": "4087fd55edd47c925a42bf0ec6653a0d92e16e67290e6cc867fc37379ed19214",
    "3407": "8eecd3e4f82131f03622d3c2d4425b576bc4e09b9c190a71d0571b33117f9812",
    "config": "7a4ca83a55f6f26e4e056b773913f5264442a5d7dc4e240636341a10362ea45d",
    "test_annotation": "130cde81a32820cd83091e78bdbce236753c7675306da004b88bf54bad130a71",
}


def sha256(path):
    result = hashlib.sha256()
    with open(str(path), "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--weights-dir", type=Path, default=PACKAGE / "weights")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    project = args.project_root.resolve()
    data = args.data_root.resolve()
    out = args.output_dir.resolve()
    checkpoints = {seed: args.weights_dir.resolve() / ("baseline_seed" + seed + "_final.pth")
                   for seed in ("3407", "42")}
    if out.exists():
        raise ValueError("Use a NEW output directory to preserve prior observations")
    # Existing train_cardd.py registers and validates all three official splits.
    for split in ("train", "val", "test"):
        for path in (data / (split + "2017"), data / "annotations" / ("instances_" + split + "2017.json")):
            if not path.exists():
                raise FileNotFoundError(str(path))
    required_hashes = {project / BASELINE: HASHES["config"],
                       data / "annotations/instances_test2017.json": HASHES["test_annotation"]}
    required_hashes.update({path: HASHES[seed] for seed, path in checkpoints.items()})
    for path, expected in required_hashes.items():
        if sha256(path) != expected:
            raise ValueError("Frozen input SHA256 mismatch: " + str(path))
    for entry in ("train_cardd.py", "offline_tools/compare_fusion_rules.py"):
        if not (project / entry).is_file():
            raise FileNotFoundError(str(project / entry))
    commands = []
    for seed in ("3407", "42"):
        commands.append(("seed" + seed, [
            sys.executable, "train_cardd.py", "--num-gpus", "1", "--eval-only",
            "--config-file", BASELINE,
            "MODEL.WEIGHTS", str(checkpoints[seed]),
            "DATASETS.TEST", "('cardd_test',)",
            "OUTPUT_DIR", str(out / ("seed" + seed)),
            "DATALOADER.NUM_WORKERS", "2",
        ]))
    commands.append(("fusion", [
        sys.executable, "offline_tools/compare_fusion_rules.py",
        "--base-results", str(out / "seed3407/inference/coco_instances_results.json"),
        "--other-results", str(out / "seed42/inference/coco_instances_results.json"),
        "--gt-json", str(data / "annotations/instances_test2017.json"),
        "--output-dir", str(out / "fusion"),
        "--match-iou", "0.5", "--max-dets", "100", "--rules", "mask_nms,consensus",
    ]))
    for name, command in commands:
        print(name + ": " + " ".join(shlex.quote(item) for item in command), flush=True)
    if args.dry_run:
        print("INPUT_HASHES_AND_COMMANDS_VALIDATED; no inference executed")
        return
    out.mkdir(parents=True)
    environment = os.environ.copy()
    environment["CARDD_ROOT"] = str(data)
    with open(str(out / "commands.json"), "w") as handle:
        json.dump({"cwd": str(project), "CARDD_ROOT": str(data), "commands": commands}, handle, indent=2)
    for name, command in commands:
        with open(str(out / (name + "_console.log")), "w") as log:
            completed = subprocess.run(command, cwd=str(project), env=environment, stdout=log, stderr=subprocess.STDOUT)
        (out / (name + "_exit_code.txt")).write_text(str(completed.returncode) + "\n")
        if completed.returncode:
            raise SystemExit("Failed " + name + "; inspect " + str(out / (name + "_console.log")))
    (out / "status.txt").write_text("COMPLETE\n")
    print("Frozen inference complete. This is a reproduction, not a new test-set tuning run.")


if __name__ == "__main__":
    main()
