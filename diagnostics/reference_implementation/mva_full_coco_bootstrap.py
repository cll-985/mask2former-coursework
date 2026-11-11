#!/usr/bin/env python3
"""Paired image-cluster bootstrap that reruns full COCOeval per resample.

Duplicate sampled images receive fresh image and annotation IDs. No per-image
AP averaging or cached matching approximation is used for the main interval.
"""
import argparse
import contextlib
import csv
import hashlib
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from offline_tools.compare_fusion_rules import atomic_json_dump


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def mask_predictions(values):
    output = []
    for source in values:
        item = dict(source)
        item.pop('bbox', None)
        item.pop('area', None)
        output.append(item)
    return output


def evaluate(dataset, predictions):
    """Return the complete COCO evaluator, with stdout silenced."""
    with open(os.devnull, 'w') as sink, contextlib.redirect_stdout(sink):
        gt = COCO()
        gt.dataset = dataset
        gt.createIndex()
        predictions = mask_predictions(predictions)
        if predictions:
            dt = gt.loadRes(predictions)
        else:
            dt = COCO()
            dt.dataset = {'images': dataset['images'],
                          'categories': dataset['categories'], 'annotations': []}
            dt.createIndex()
        evaluator = COCOeval(gt, dt, 'segm')
        evaluator.evaluate()
        evaluator.accumulate()
    return evaluator


def full_ap(dataset, predictions):
    precision = evaluate(dataset, predictions).eval['precision'][:, :, :, 0, -1]
    valid = precision[precision > -1]
    if not valid.size:
        raise ValueError('No valid GT categories in full COCO evaluation')
    return float(np.mean(valid) * 100.0)


def group(values):
    groups = defaultdict(list)
    for value in values:
        groups[int(value['image_id'])].append(value)
    return groups


class Resampler:
    def __init__(self, dataset, systems):
        self.dataset = dataset
        self.images = {int(x['id']): x for x in dataset['images']}
        self.ids = sorted(self.images)
        if len(self.ids) != len(dataset['images']):
            raise ValueError('Repeated source image IDs')
        if len({int(x['id']) for x in dataset['annotations']}) != len(dataset['annotations']):
            raise ValueError('Repeated source annotation IDs')
        self.annotations = group(dataset['annotations'])
        self.systems = [group(x) for x in systems]
        allowed = set(self.ids)
        for values in systems:
            if any(int(x['image_id']) not in allowed for x in values):
                raise ValueError('Prediction outside ground-truth image IDs')
        self.category_presence = defaultdict(set)
        for annotation in dataset['annotations']:
            if not annotation.get('iscrowd', 0):
                self.category_presence[int(annotation['image_id'])].add(int(annotation['category_id']))
        self.target_categories = set().union(*self.category_presence.values())

    def complete_categories(self, indices):
        present = set()
        for index in indices:
            present.update(self.category_presence[self.ids[int(index)]])
        return present == self.target_categories

    def draw(self, indices):
        dataset = {'info': dict(self.dataset.get('info', {})),
                   'categories': self.dataset['categories'], 'images': [], 'annotations': []}
        systems = [[] for _ in self.systems]
        for fresh_id, index in enumerate(indices, 1):
            old_id = self.ids[int(index)]
            image = dict(self.images[old_id])
            image['id'] = fresh_id
            dataset['images'].append(image)
            for source in self.annotations[old_id]:
                annotation = dict(source)
                annotation['image_id'] = fresh_id
                annotation['id'] = len(dataset['annotations']) + 1
                dataset['annotations'].append(annotation)
            for output, grouped in zip(systems, self.systems):
                for source in grouped[old_id]:
                    item = dict(source)
                    item['image_id'] = fresh_id
                    output.append(item)
        return dataset, systems


def interval(deltas):
    values = np.asarray(deltas, dtype=np.float64)
    if not values.size:
        return {'n': 0, 'ci95_low': None, 'ci95_high': None}
    return {'n': int(values.size), 'mean_delta': float(np.mean(values)),
            'ci95_low': float(np.percentile(values, 2.5)),
            'ci95_high': float(np.percentile(values, 97.5)),
            'fraction_strictly_positive': float(np.mean(values > 0.0)),
            'fraction_zero': float(np.mean(values == 0.0))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ('gt-json', 'baseline-results', 'candidate-results', 'output-dir'):
        parser.add_argument('--' + field, required=True)
    parser.add_argument('--resamples', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=20261015)
    parser.add_argument('--expected-baseline-ap', type=float)
    parser.add_argument('--expected-candidate-ap', type=float)
    args = parser.parse_args()
    if args.resamples <= 0 or os.path.exists(args.output_dir):
        raise ValueError('Require positive resamples and a new output directory')
    with open(args.gt_json) as handle:
        dataset = json.load(handle)
    with open(args.baseline_results) as handle:
        baseline = json.load(handle)
    with open(args.candidate_results) as handle:
        candidate = json.load(handle)
    resampler = Resampler(dataset, [baseline, candidate])
    group_fields = {k for image in dataset['images'] for k in
                    ('vehicle_id', 'sequence_id', 'video_id', 'group_id') if k in image}
    if group_fields:
        raise ValueError('Group metadata found: {}; freeze group-resampling first'.format(sorted(group_fields)))
    os.makedirs(args.output_dir)
    rng = np.random.RandomState(args.seed)
    # Only category presence is inspected in this audit, never prediction AP.
    missing = sum(not resampler.complete_categories(rng.randint(
        0, len(resampler.ids), size=len(resampler.ids))) for _ in range(2000))
    protocol = vars(args).copy()
    protocol.update({'estimand': 'dataset segm COCO AP delta conditional on fixed weights and empirical image distribution',
                     'units': 'AP points, 0-100', 'image_count': len(resampler.ids),
                     'category_missingness_audit_draws': 2000,
                     'category_missingness_audit_rejections': missing,
                     'group_metadata': 'no recognized group IDs; residual image dependence unknown',
                     'ci': 'paired image-cluster percentile 95%; no seed population inference',
                     'unfiltered_sensitivity': 'COCO excludes categories absent in a resample; estimand differs when missing',
                     'source_sha256': {p: sha256(p) for p in (args.gt_json, args.baseline_results, args.candidate_results)}})
    atomic_json_dump(protocol, os.path.join(args.output_dir, 'protocol.json'), indent=2)
    if missing > 20:
        raise RuntimeError('GT-only missingness >1%; freeze stratified protocol before AP analysis')
    base_ap, cand_ap = full_ap(dataset, baseline), full_ap(dataset, candidate)
    for actual, expected in ((base_ap, args.expected_baseline_ap), (cand_ap, args.expected_candidate_ap)):
        if expected is not None and abs(actual - expected) >= 1e-6:
            raise AssertionError('Historical AP mismatch: {} != {}'.format(actual, expected))
    print('ORIGINAL_AP baseline={:.10f} candidate={:.10f}'.format(base_ap, cand_ap), flush=True)
    all_deltas, complete_deltas = [], []
    rng = np.random.RandomState(args.seed)
    started = time.monotonic()
    with open(os.path.join(args.output_dir, 'resamples.csv'), 'w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['draw', 'baseline_ap', 'candidate_ap', 'delta', 'complete_categories'])
        for index in range(args.resamples):
            indices = rng.randint(0, len(resampler.ids), size=len(resampler.ids))
            complete = resampler.complete_categories(indices)
            sampled_gt, sampled_predictions = resampler.draw(indices)
            ap0, ap1 = [full_ap(sampled_gt, x) for x in sampled_predictions]
            delta = ap1 - ap0
            all_deltas.append(delta)
            if complete:
                complete_deltas.append(delta)
            writer.writerow([index, ap0, ap1, delta, int(complete)])
            if (index + 1) % 10 == 0 or index + 1 == args.resamples:
                handle.flush()
                progress = {'completed': index + 1, 'target': args.resamples,
                            'elapsed_seconds': time.monotonic() - started,
                            'complete_category_resamples': len(complete_deltas),
                            'partial_ci_must_not_be_used_for_stopping': True}
                atomic_json_dump(progress, os.path.join(args.output_dir, 'progress.json'), indent=2)
                print('BOOTSTRAP_PROGRESS {}'.format(json.dumps(progress)), flush=True)
    report = {'protocol': protocol, 'original_baseline_ap': base_ap,
              'original_candidate_ap': cand_ap, 'original_delta': cand_ap - base_ap,
              'complete_category_conditional_interval': interval(complete_deltas),
              'unfiltered_sensitivity_interval': interval(all_deltas),
              'status': 'complete' if args.resamples >= 2000 else 'smoke_only',
              'elapsed_seconds': time.monotonic() - started}
    atomic_json_dump(report, os.path.join(args.output_dir, 'full_coco_bootstrap_report.json'), indent=2)
    print('FULL_COCO_BOOTSTRAP_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
