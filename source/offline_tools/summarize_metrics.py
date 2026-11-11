#!/usr/bin/env python3
"""Summarize Detectron2 JSON-lines metrics without extra dependencies."""

from __future__ import print_function

import argparse
import json
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if not os.path.isfile(args.metrics):
        raise SystemExit("Metrics file does not exist: {0}".format(args.metrics))

    records = []
    with open(args.metrics, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except ValueError:
                print("Skipping invalid JSON line {0}".format(line_number))

    if not records:
        raise SystemExit("No valid records found")

    metric_names = set()
    for record in records:
        metric_names.update(record.keys())
    interesting = sorted(name for name in metric_names if (
        name.startswith("segm/") or name.startswith("bbox/") or
        name.startswith("validation_loss") or name in ("total_loss", "lr", "iteration")
    ))

    summary = {"metrics_file": os.path.abspath(args.metrics), "records": len(records), "metrics": {}}
    for name in interesting:
        values = [(record.get("iteration"), record[name]) for record in records if isinstance(record.get(name), (int, float))]
        if not values:
            continue
        last_iteration, last_value = values[-1]
        if name in ("iteration", "lr"):
            summary["metrics"][name] = {
                "last": last_value,
                "last_iteration": last_iteration,
            }
            continue
        maximize = name.startswith("segm/") or name.startswith("bbox/")
        best = max(values, key=lambda item: item[1]) if maximize else min(values, key=lambda item: item[1])
        summary["metrics"][name] = {
            "last": last_value,
            "last_iteration": last_iteration,
            "best": best[1],
            "best_iteration": best[0],
        }

    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print("Wrote {0}".format(os.path.abspath(args.output)))


if __name__ == "__main__":
    main()
