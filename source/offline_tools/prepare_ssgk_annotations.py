#!/usr/bin/env python3
"""Create publication-safe 9-class COCO manifests for SSGK.

The source JSON contains an unused category id 0 named ``_background_``.
Detectron2 would otherwise allocate a learnable class for it.  This script
removes only that category definition, verifies that no annotation uses it,
and preserves every image and foreground annotation verbatim.
"""

from __future__ import print_function

import argparse
import hashlib
import json
import os


EXPECTED_FOREGROUND = {
    1: "Dislocation",
    2: "Glass_breakage",
    3: "Glass_crack",
    4: "Missing",
    5: "Mild_deformation",
    6: "Severe_deformation",
    7: "Scratch",
    8: "Scratches",
    9: "Tearing",
}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def convert(source, destination):
    with open(source, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    categories = {int(item["id"]): item["name"] for item in data["categories"]}
    if categories.get(0) not in ("_background_", "background"):
        raise ValueError("Expected unused background category id 0 in {0}".format(source))
    foreground = {key: value for key, value in categories.items() if key != 0}
    if foreground != EXPECTED_FOREGROUND:
        raise ValueError(
            "Unexpected foreground taxonomy in {0}: {1}".format(source, foreground)
        )
    background_annotations = [
        annotation for annotation in data["annotations"]
        if int(annotation["category_id"]) == 0
    ]
    if background_annotations:
        raise ValueError(
            "Refusing to drop {0} annotations assigned to category 0".format(
                len(background_annotations)
            )
        )

    data["categories"] = [
        category for category in data["categories"] if int(category["id"]) != 0
    ]
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    with open(destination, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, separators=(",", ":"))
    return {
        "source": os.path.abspath(source),
        "source_sha256": sha256(source),
        "destination": os.path.abspath(destination),
        "destination_sha256": sha256(destination),
        "images": len(data["images"]),
        "annotations": len(data["annotations"]),
        "categories": len(data["categories"]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default=os.environ.get(
            "SSGK_ROOT",
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "datasets", "coco_ssgk"),
        ),
    )
    args = parser.parse_args()
    root = os.path.abspath(args.root)
    records = []
    for split in ("train2017", "val2017"):
        records.append(
            convert(
                os.path.join(root, "annotations", "instances_{0}.json".format(split)),
                os.path.join(root, "annotations_9cls", "instances_{0}.json".format(split)),
            )
        )
    manifest = os.path.join(root, "annotations_9cls", "manifest.json")
    with open(manifest, "w", encoding="utf-8") as handle:
        json.dump({"transformation": "remove_unused_category_0", "files": records}, handle,
                  ensure_ascii=False, indent=2, sort_keys=True)
    print("Wrote {0}".format(manifest))
    for record in records:
        print(
            "images={images} annotations={annotations} categories={categories} "
            "destination={destination}".format(**record)
        )


if __name__ == "__main__":
    main()

