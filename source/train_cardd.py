#!/usr/bin/env python3
"""Reproducible CarDD entry point for Mask2Former.

The dataset root defaults to <project>/CarDD_COCO and can be overridden with
CARDD_ROOT. Registration is performed before the regular Mask2Former trainer
builds its config and data loaders.
"""

from __future__ import print_function

import os

from detectron2.data import DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import default_argument_parser, launch

from train_net import main


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CARDD_ROOT = os.path.abspath(
    os.environ.get("CARDD_ROOT", os.path.join(PROJECT_ROOT, "CarDD_COCO"))
)


CARDD_SPLITS = {
    "cardd_train": (
        os.path.join(CARDD_ROOT, "annotations", "instances_train2017.json"),
        os.path.join(CARDD_ROOT, "train2017"),
    ),
    "cardd_val": (
        os.path.join(CARDD_ROOT, "annotations", "instances_val2017.json"),
        os.path.join(CARDD_ROOT, "val2017"),
    ),
    "cardd_test": (
        os.path.join(CARDD_ROOT, "annotations", "instances_test2017.json"),
        os.path.join(CARDD_ROOT, "test2017"),
    ),
}


def register_cardd():
    missing = []
    for dataset_name, (json_file, image_root) in CARDD_SPLITS.items():
        if not os.path.isfile(json_file):
            missing.append(json_file)
        if not os.path.isdir(image_root):
            missing.append(image_root)
        if dataset_name not in DatasetCatalog.list():
            register_coco_instances(dataset_name, {}, json_file, image_root)
    if missing:
        raise FileNotFoundError(
            "CarDD is incomplete under {0}. Missing:\n{1}".format(
                CARDD_ROOT, "\n".join("  - " + path for path in missing)
            )
        )


if __name__ == "__main__":
    register_cardd()
    args = default_argument_parser().parse_args()
    print("CarDD root:", CARDD_ROOT)
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
