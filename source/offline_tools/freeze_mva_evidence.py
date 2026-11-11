#!/usr/bin/env python3
"""Create a read-only, SHA-256 evidence inventory using Python stdlib only.

No training, evaluation, source outputs, or checkpoints are modified. Existing
checkpoint hashes are preserved in source reports; large checkpoints are not
rehash targets. File timestamps establish availability only, not preregistration.
"""
from __future__ import print_function

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path


SUMMARY_NAMES = {
    'metrics_summary.json', 'protocol.json', 'status.txt', 'exit_code.txt',
    'fusion_rule_comparison.json', 'paired_bootstrap_report.json',
    'paired_bootstrap_summary.csv', 'fusion_metrics_corrected_mask_area.csv',
    'fusion_metrics_corrected_mask_area.json', 'selected_cases.json',
    'soup_report.json', 'interpolation_report.json',
}


def inside(path, root):
    try:
        path.resolve().relative_to(root)
        return True
    except (ValueError, OSError):
        return False


def digest(path):
    before = path.stat()
    checksum = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            checksum.update(block)
    after = path.stat()
    return checksum.hexdigest(), (before.st_size == after.st_size and
                                  before.st_mtime_ns == after.st_mtime_ns)


def selected(path):
    name = path.name
    return (name in SUMMARY_NAMES or name.endswith('coco_instances_results.json')
            or name.startswith('layerwise_summary_')
            or (name.endswith('.json') and 'P5_fixed_soup_R50' in str(path)))


def collect(root):
    paths = set()
    for folder in ('output_offline', 'refine-logs', 'experiment_records'):
        parent = root / folder
        if not parent.is_dir():
            continue
        for directory, subdirs, names in os.walk(str(parent), followlinks=False):
            subdirs[:] = sorted(d for d in subdirs if not d.startswith('.'))
            for name in sorted(names):
                path = Path(directory) / name
                if folder == 'output_offline' and not selected(path):
                    continue
                if folder != 'output_offline' and path.suffix not in (
                        '.md', '.json', '.csv', '.txt'):
                    continue
                paths.add(path)
    for pattern in ('offline_tools/*fusion*.py', 'offline_tools/*bootstrap*.py',
                    'offline_tools/fuse_coco_instances.py',
                    'offline_tools/freeze_mva_evidence.py',
                    'offline_tools/experiment_registry.csv', 'PAPER_PLAN.md',
                    'CarDD_COCO/annotations/*.json'):
        paths.update(root.glob(pattern))
    artifacts, skipped = [], []
    for path in sorted(paths, key=lambda p: str(p)):
        relative = str(path.relative_to(root)).replace(os.sep, '/')
        if not inside(path, root):
            skipped.append({'path': relative, 'reason': 'resolves outside project'})
            continue
        if not path.is_file():
            continue
        sha, stable = digest(path)
        stat = path.stat()
        item = {'path': relative, 'size_bytes': stat.st_size, 'sha256': sha,
                'stable_during_hash': stable,
                'mtime_utc': datetime.datetime.fromtimestamp(
                    stat.st_mtime, datetime.timezone.utc).isoformat()}
        if path.suffix == '.json' and stat.st_size < 4 * 1024 * 1024 and not (
                name_is_prediction(path.name) or '/annotations/' in relative):
            try:
                with path.open('r', encoding='utf-8') as handle:
                    item['source_payload'] = json.load(handle)
            except (ValueError, UnicodeError) as exc:
                item['parse_error'] = str(exc)
        artifacts.append(item)
    return artifacts, skipped


def name_is_prediction(name):
    return name.endswith('coco_instances_results.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', default='.')
    parser.add_argument('--output-dir', required=True,
                        help='New project-relative directory; must not exist')
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    output = root / args.output_dir
    if not inside(output, root):
        parser.error('output-dir must resolve inside project')
    if output.exists():
        parser.error('output directory already exists; choose a new run name')
    artifacts, skipped = collect(root)
    unstable = [x['path'] for x in artifacts if not x['stable_during_hash']]
    tz = datetime.timezone(datetime.timedelta(hours=8))
    report = {
        'schema_version': 1, 'created_at_asia_shanghai':
        datetime.datetime.now(tz).isoformat(), 'project_root': str(root),
        'artifact_count': len(artifacts), 'artifacts': artifacts,
        'skipped': skipped, 'unstable_sources': unstable,
        'scope': 'Selected historical summaries, predictions, protocols, tools, annotations',
        'limitations': [
            'Inventory only: no new AP or confidence intervals are computed.',
            'Timestamps/hashes do not establish preregistration or first data access.',
            'Checkpoints are not rehashed; existing checkpoint hashes remain in reports.',
            'Symlinks resolving outside the authorized project are skipped.',
            'Independent seed pairs share models and are not independent replicates.',
        ],
    }
    output.mkdir(parents=True)
    target = output / 'evidence_snapshot.json'
    with target.open('x', encoding='utf-8') as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write('\n')
    with (output / 'SHA256SUMS.txt').open('x', encoding='utf-8') as handle:
        for item in artifacts:
            handle.write('{}  {}\n'.format(item['sha256'], item['path']))
    print(json.dumps({'output': str(target), 'artifacts': len(artifacts),
                      'skipped': len(skipped), 'unstable': len(unstable)}, sort_keys=True))
    if unstable:
        raise SystemExit('Sources changed during hashing; snapshot requires review')


if __name__ == '__main__':
    main()
