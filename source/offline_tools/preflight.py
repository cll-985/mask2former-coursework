#!/usr/bin/env python3
"""Preflight checks for an offline Mask2Former training container."""

from __future__ import print_function

import argparse
import importlib
import os
import platform
import shutil
import sys


def import_version(name):
    try:
        module = importlib.import_module(name)
        return True, getattr(module, "__version__", "unknown")
    except Exception as error:
        return False, "{0}: {1}".format(type(error).__name__, error)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--required-file", action="append", default=[])
    args = parser.parse_args()

    failures = []
    warnings = []
    print("python={0}".format(sys.version.replace("\n", " ")))
    print("platform={0}".format(platform.platform()))

    modules = ["torch", "torchvision", "detectron2", "cv2", "pycocotools"]
    loaded = {}
    for module_name in modules:
        ok, version = import_version(module_name)
        loaded[module_name] = ok
        print("module.{0}={1}".format(module_name, version))
        if not ok:
            failures.append("cannot import {0}".format(module_name))

    if loaded.get("torch"):
        import torch
        print("torch.cuda_available={0}".format(torch.cuda.is_available()))
        print("torch.cuda_device_count={0}".format(torch.cuda.device_count()))
        if torch.cuda.is_available():
            for index in range(torch.cuda.device_count()):
                print("torch.cuda_device_{0}={1}".format(index, torch.cuda.get_device_name(index)))
        else:
            failures.append("CUDA is not available to PyTorch")

    config = os.path.abspath(args.config)
    if not os.path.isfile(config):
        failures.append("config does not exist: {0}".format(config))
    else:
        print("config={0}".format(config))
        with open(config, "r", encoding="utf-8") as handle:
            config_text = handle.read().lower()
        if "http://" in config_text or "https://" in config_text or "detectron2://" in config_text:
            warnings.append("config contains a remote weight/resource URL; replace it with a verified local path")

    for required in args.required_file:
        absolute = os.path.abspath(required)
        exists = os.path.isfile(absolute)
        print("required_file.{0}={1}".format(absolute, exists))
        if not exists:
            failures.append("required file missing: {0}".format(absolute))

    disk_root = os.path.abspath(os.curdir)
    usage = shutil.disk_usage(disk_root)
    print("disk_free_gib={0:.2f}".format(usage.free / float(1024 ** 3)))
    if usage.free < 20 * 1024 ** 3:
        warnings.append("less than 20 GiB free space")

    for warning in warnings:
        print("WARNING: {0}".format(warning))
    for failure in failures:
        print("ERROR: {0}".format(failure))
    if failures:
        raise SystemExit(1)
    print("PREFLIGHT_OK")


if __name__ == "__main__":
    main()
