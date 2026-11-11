#!/usr/bin/env python3
"""Create a compact ZIP of a run for transfer from an offline server."""

from __future__ import print_function

import argparse
import hashlib
import json
import os
import zipfile


DEFAULT_EXTENSIONS = {
    ".json", ".yaml", ".yml", ".txt", ".log", ".csv", ".png", ".jpg", ".jpeg", ".html"
}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--include-checkpoints", action="store_true")
    args = parser.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    output = os.path.abspath(args.output)
    if not os.path.isdir(run_dir):
        raise SystemExit("Run directory does not exist: {0}".format(run_dir))

    selected = []
    for root, _, files in os.walk(run_dir):
        for filename in files:
            path = os.path.join(root, filename)
            extension = os.path.splitext(filename)[1].lower()
            if extension in DEFAULT_EXTENSIONS or (args.include_checkpoints and extension in (".pth", ".pkl")):
                selected.append(path)

    manifest = []
    output_dir = os.path.dirname(output)
    if output_dir and not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(selected):
            relative = os.path.relpath(path, run_dir)
            archive.write(path, relative)
            manifest.append({"path": relative.replace(os.sep, "/"), "bytes": os.path.getsize(path), "sha256": sha256(path)})
        archive.writestr("transfer_manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))

    print("Packed {0} files into {1}".format(len(selected), output))
    if not args.include_checkpoints:
        print("Checkpoints were excluded; pass --include-checkpoints when needed.")


if __name__ == "__main__":
    main()
