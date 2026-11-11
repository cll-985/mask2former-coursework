#!/usr/bin/env python3
"""Reproducible entry point for the trusted SSGK damage dataset.

The dataset root defaults to ``<project>/datasets/coco_ssgk`` and can be
overridden with ``SSGK_ROOT``.  Publication runs use derived COCO annotations
under ``annotations_9cls`` where the unused background category is removed;
the original images and annotations are never modified.
"""

from __future__ import print_function

import os

from detectron2.data import DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import default_argument_parser, launch

from train_net import main


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SSGK_ROOT = os.path.abspath(
    os.environ.get("SSGK_ROOT", os.path.join(PROJECT_ROOT, "datasets", "coco_ssgk"))
)
ANNOTATION_ROOT = os.path.join(SSGK_ROOT, "annotations_9cls")

SSGK_SPLITS = {
    "ssgk_train": (
        os.path.join(ANNOTATION_ROOT, "instances_train2017.json"),
        os.path.join(SSGK_ROOT, "train2017"),
    ),
    "ssgk_val": (
        os.path.join(ANNOTATION_ROOT, "instances_val2017.json"),
        os.path.join(SSGK_ROOT, "val2017"),
    ),
}


def register_ssgk():
    missing = []
    for dataset_name, (json_file, image_root) in SSGK_SPLITS.items():
        if not os.path.isfile(json_file):
            missing.append(json_file)
        if not os.path.isdir(image_root):
            missing.append(image_root)
        if dataset_name not in DatasetCatalog.list():
            register_coco_instances(dataset_name, {}, json_file, image_root)
    if missing:
        raise FileNotFoundError(
            "SSGK is incomplete under {0}. Missing:\n{1}\n"
            "Run: python offline_tools/prepare_ssgk_annotations.py".format(
                SSGK_ROOT, "\n".join("  - " + path for path in missing)
            )
        )


if __name__ == "__main__":
    register_ssgk()
    args = default_argument_parser().parse_args()
    print("SSGK root:", SSGK_ROOT)
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )

