#!/usr/bin/env python3
"""Audit one or more COCO annotation files without third-party packages."""

from __future__ import print_function

import argparse
import csv
import json
import math
import os


def polygon_perimeter(segmentation):
    total = 0.0
    valid = False
    if not isinstance(segmentation, list):
        return None
    for polygon in segmentation:
        if not isinstance(polygon, list) or len(polygon) < 6 or len(polygon) % 2:
            continue
        points = list(zip(polygon[0::2], polygon[1::2]))
        for index, point in enumerate(points):
            next_point = points[(index + 1) % len(points)]
            total += math.hypot(point[0] - next_point[0], point[1] - next_point[1])
        valid = True
    return total if valid else None


def size_name(area):
    if area < 32.0 * 32.0:
        return "small"
    if area < 96.0 * 96.0:
        return "medium"
    return "large"


def parse_spec(spec):
    if "=" not in spec:
        raise ValueError("annotation must be NAME=PATH: {0}".format(spec))
    name, path = spec.split("=", 1)
    return name.strip(), os.path.abspath(path.strip())


def audit(name, path):
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    images = data.get("images", [])
    annotations = data.get("annotations", [])
    categories = data.get("categories", [])
    image_ids = [item.get("id") for item in images]
    annotation_ids = [item.get("id") for item in annotations]
    category_ids = [item.get("id") for item in categories]
    image_id_set = set(image_ids)
    category_id_set = set(category_ids)
    category_name = {item.get("id"): item.get("name", str(item.get("id"))) for item in categories}

    class_stats = {}
    size_stats = {"small": 0, "medium": 0, "large": 0}
    compactness_values = []
    errors = {
        "duplicate_image_ids": len(image_ids) - len(image_id_set),
        "duplicate_annotation_ids": len(annotation_ids) - len(set(annotation_ids)),
        "missing_image_reference": 0,
        "unknown_category_reference": 0,
        "invalid_bbox": 0,
        "nonpositive_area": 0,
        "empty_segmentation": 0,
    }

    for annotation in annotations:
        category_id = annotation.get("category_id")
        label = category_name.get(category_id, "UNKNOWN_{0}".format(category_id))
        stats = class_stats.setdefault(label, {
            "instances": 0,
            "area_sum": 0.0,
            "small": 0,
            "medium": 0,
            "large": 0,
            "compactness_sum": 0.0,
            "compactness_count": 0,
        })
        stats["instances"] += 1

        if annotation.get("image_id") not in image_id_set:
            errors["missing_image_reference"] += 1
        if category_id not in category_id_set:
            errors["unknown_category_reference"] += 1

        bbox = annotation.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4 or bbox[2] <= 0 or bbox[3] <= 0:
            errors["invalid_bbox"] += 1

        area = float(annotation.get("area") or 0.0)
        if area <= 0:
            errors["nonpositive_area"] += 1
            if isinstance(bbox, list) and len(bbox) == 4:
                area = max(float(bbox[2]), 0.0) * max(float(bbox[3]), 0.0)
        stats["area_sum"] += area
        size = size_name(area)
        stats[size] += 1
        size_stats[size] += 1

        segmentation = annotation.get("segmentation")
        if segmentation in (None, [], {}):
            errors["empty_segmentation"] += 1
        perimeter = polygon_perimeter(segmentation)
        if perimeter and area > 0:
            compactness = 4.0 * math.pi * area / (perimeter * perimeter + 1e-12)
            compactness_values.append(compactness)
            stats["compactness_sum"] += compactness
            stats["compactness_count"] += 1

    image_areas = [float(item.get("width", 0)) * float(item.get("height", 0)) for item in images]
    instances_per_image = float(len(annotations)) / len(images) if images else 0.0
    compactness_values.sort()

    def quantile(values, fraction):
        if not values:
            return None
        position = fraction * (len(values) - 1)
        low = int(math.floor(position))
        high = int(math.ceil(position))
        if low == high:
            return values[low]
        return values[low] + (values[high] - values[low]) * (position - low)

    class_rows = []
    for label in sorted(class_stats):
        stats = class_stats[label]
        class_rows.append({
            "split": name,
            "class": label,
            "instances": stats["instances"],
            "small": stats["small"],
            "medium": stats["medium"],
            "large": stats["large"],
            "mean_area": stats["area_sum"] / stats["instances"] if stats["instances"] else None,
            "mean_polygon_compactness": (
                stats["compactness_sum"] / stats["compactness_count"]
                if stats["compactness_count"] else None
            ),
        })

    return {
        "split": name,
        "path": path,
        "images": len(images),
        "annotations": len(annotations),
        "categories": [{"id": item.get("id"), "name": item.get("name")} for item in categories],
        "instances_per_image": instances_per_image,
        "mean_image_area": sum(image_areas) / len(image_areas) if image_areas else None,
        "size_counts": size_stats,
        "polygon_compactness_quantiles": {
            "q25": quantile(compactness_values, 0.25),
            "q50": quantile(compactness_values, 0.50),
            "q75": quantile(compactness_values, 0.75),
            "count": len(compactness_values),
        },
        "errors": errors,
        "class_rows": class_rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation", action="append", required=True, help="NAME=PATH; repeat per split")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output_dir = os.path.abspath(args.output_dir)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    reports = []
    class_rows = []
    for spec in args.annotation:
        name, path = parse_spec(spec)
        report = audit(name, path)
        class_rows.extend(report.pop("class_rows"))
        reports.append(report)

    json_path = os.path.join(output_dir, "coco_audit.json")
    csv_path = os.path.join(output_dir, "coco_class_stats.csv")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump({"splits": reports}, handle, ensure_ascii=False, indent=2)
    fields = ["split", "class", "instances", "small", "medium", "large", "mean_area", "mean_polygon_compactness"]
    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(class_rows)

    print("Wrote {0}".format(json_path))
    print("Wrote {0}".format(csv_path))
    for report in reports:
        print("{0}: images={1}, instances={2}, sizes={3}, errors={4}".format(
            report["split"], report["images"], report["annotations"],
            report["size_counts"], report["errors"]))


if __name__ == "__main__":
    main()
