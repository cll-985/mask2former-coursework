#!/usr/bin/env python3
"""Create a fixed-ratio model soup from two compatible checkpoints.

Only floating-point model tensors are interpolated. Integer/bool buffers must
match exactly and are copied unchanged. Optimizer/trainer state is deliberately
excluded because the output checkpoint is intended for evaluation.
"""

from __future__ import print_function

import argparse
import hashlib
import json
import os

import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--other", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Output = (1-alpha)*base + alpha*other for floating tensors.",
    )
    return parser.parse_args()


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_model(path):
    checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError("checkpoint has no top-level 'model': {}".format(path))
    if not isinstance(checkpoint["model"], dict):
        raise TypeError("checkpoint 'model' is not a mapping: {}".format(path))
    return checkpoint["model"]


def interpolate_models(base_model, other_model, alpha):
    base_keys = list(base_model.keys())
    other_keys = list(other_model.keys())
    if base_keys != other_keys:
        missing = sorted(set(base_keys) - set(other_keys))
        extra = sorted(set(other_keys) - set(base_keys))
        raise ValueError(
            "model keys differ; missing={} extra={}".format(missing, extra)
        )

    output = type(base_model)()
    floating_tensors = 0
    fixed_tensors = 0
    max_endpoint_delta = 0.0

    with torch.no_grad():
        for key in base_keys:
            base_value = base_model[key]
            other_value = other_model[key]
            if not torch.is_tensor(base_value) or not torch.is_tensor(other_value):
                if base_value != other_value:
                    raise ValueError("non-tensor value differs for key {}".format(key))
                output[key] = base_value
                continue
            if base_value.shape != other_value.shape:
                raise ValueError(
                    "shape differs for {}: {} vs {}".format(
                        key, tuple(base_value.shape), tuple(other_value.shape)
                    )
                )
            if base_value.dtype != other_value.dtype:
                raise ValueError(
                    "dtype differs for {}: {} vs {}".format(
                        key, base_value.dtype, other_value.dtype
                    )
                )

            if torch.is_floating_point(base_value):
                mixed = base_value.float().mul(1.0 - alpha)
                mixed.add_(other_value.float(), alpha=alpha)
                output[key] = mixed.to(dtype=base_value.dtype)
                floating_tensors += 1
                if base_value.numel() > 0:
                    delta = (base_value.float() - other_value.float()).abs().max()
                    max_endpoint_delta = max(max_endpoint_delta, float(delta.item()))
            else:
                if not torch.equal(base_value, other_value):
                    raise ValueError("non-floating tensor differs for key {}".format(key))
                output[key] = base_value.clone()
                fixed_tensors += 1

    return output, {
        "floating_tensors": floating_tensors,
        "fixed_tensors": fixed_tensors,
        "max_endpoint_abs_delta": max_endpoint_delta,
    }


def main():
    args = parse_args()
    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")

    base_path = os.path.abspath(args.base)
    other_path = os.path.abspath(args.other)
    output_path = os.path.abspath(args.output)
    if output_path in (base_path, other_path):
        raise ValueError("output must not overwrite an input checkpoint")

    base_model = load_model(base_path)
    other_model = load_model(other_path)
    mixed_model, stats = interpolate_models(base_model, other_model, args.alpha)

    metadata = {
        "method": "fixed_checkpoint_interpolation",
        "alpha": args.alpha,
        "formula": "(1-alpha)*base + alpha*other",
        "base": base_path,
        "other": other_path,
        "base_sha256": file_sha256(base_path),
        "other_sha256": file_sha256(other_path),
        "stats": stats,
    }
    checkpoint = {"model": mixed_model, "iteration": -1, "soup": metadata}

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    temporary_path = output_path + ".tmp"
    torch.save(checkpoint, temporary_path)
    os.replace(temporary_path, output_path)

    metadata["output"] = output_path
    metadata["output_sha256"] = file_sha256(output_path)
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
