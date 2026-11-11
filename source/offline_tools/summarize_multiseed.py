#!/usr/bin/env python3
"""Aggregate final Mask2Former metrics across reproducibility seeds."""

from __future__ import print_function

import argparse
import csv
import json
import os
import statistics


METRICS = (
    "segm/AP", "segm/AP50", "segm/AP75", "segm/APs", "segm/APm", "segm/APl",
    "segm/AP-dent", "segm/AP-scratch", "segm/AP-crack",
    "segm/AP-glass shatter", "segm/AP-lamp broken", "segm/AP-tire flat",
)


def parse_run(value):
    if "=" not in value:
        raise argparse.ArgumentTypeError("run must be SEED=RUN_DIR")
    seed, run_dir = value.split("=", 1)
    if not seed or not run_dir:
        raise argparse.ArgumentTypeError("run must be SEED=RUN_DIR")
    return seed, os.path.abspath(run_dir)


def load_run(seed, run_dir):
    summary_path = os.path.join(run_dir, "metrics_summary.json")
    exit_path = os.path.join(run_dir, "exit_code.txt")
    model_path = os.path.join(run_dir, "model_final.pth")
    with open(summary_path, "r") as handle:
        summary = json.load(handle)
    metrics = summary["metrics"]
    iteration = int(metrics["iteration"]["last"])
    exit_code = None
    if os.path.isfile(exit_path):
        with open(exit_path, "r") as handle:
            exit_code = handle.read().strip()
    complete = exit_code == "0" and os.path.getsize(model_path) > 0 and iteration == 368750
    return {
        "seed": seed,
        "run_dir": run_dir,
        "complete": complete,
        "iteration": iteration,
        "values": {name: float(metrics[name]["last"]) for name in METRICS},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", required=True, type=parse_run,
                        help="SEED=RUN_DIR; repeat once per seed")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    runs = [load_run(seed, run_dir) for seed, run_dir in args.run]
    if len(runs) < 2:
        raise ValueError("At least two runs are required")
    incomplete = [run["seed"] for run in runs if not run["complete"]]
    if incomplete:
        raise RuntimeError("Incomplete runs: " + ", ".join(incomplete))

    os.makedirs(args.output_dir, exist_ok=True)
    rows = []
    for run in runs:
        row = {"seed": run["seed"], "run_dir": run["run_dir"]}
        row.update(run["values"])
        rows.append(row)

    aggregate = {}
    for name in METRICS:
        values = [run["values"][name] for run in runs]
        aggregate[name] = {
            "n": len(values),
            "mean": statistics.mean(values),
            "sample_sd": statistics.stdev(values),
            "min": min(values),
            "max": max(values),
            "range": max(values) - min(values),
        }

    csv_path = os.path.join(args.output_dir, "baseline_multiseed_raw.csv")
    with open(csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["seed", "run_dir"] + list(METRICS))
        writer.writeheader()
        writer.writerows(rows)

    json_path = os.path.join(args.output_dir, "baseline_multiseed_summary.json")
    with open(json_path, "w") as handle:
        json.dump({"runs": rows, "aggregate": aggregate}, handle, indent=2, sort_keys=True)

    md_path = os.path.join(args.output_dir, "baseline_multiseed_table.md")
    display = METRICS[:6]
    with open(md_path, "w") as handle:
        handle.write("| Seed | AP | AP50 | AP75 | APs | APm | APl |\n")
        handle.write("|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in rows:
            handle.write("| {} | {} |\n".format(
                row["seed"], " | ".join("{:.4f}".format(row[name]) for name in display)))
        handle.write("| mean±SD | {} |\n".format(" | ".join(
            "{:.4f}±{:.4f}".format(aggregate[name]["mean"], aggregate[name]["sample_sd"])
            for name in display)))

    print("Wrote", csv_path)
    print("Wrote", json_path)
    print("Wrote", md_path)


if __name__ == "__main__":
    main()
