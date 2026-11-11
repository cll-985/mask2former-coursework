#!/usr/bin/env python3
"""Frozen TP/FP changes and GT-assisted proposal-recall diagnostics."""
import argparse
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np
from pycocotools import mask as mask_util

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from offline_tools.compare_fusion_rules import atomic_json_dump
from offline_tools.fuse_coco_instances import rle_for_compute
from offline_tools.mva_full_coco_bootstrap import evaluate


def fingerprint(prediction):
    return (int(prediction['image_id']), int(prediction['category_id']),
            json.dumps(prediction['segmentation'], sort_keys=True))


def matching_summary(evaluator, threshold):
    index = int(np.flatnonzero(np.isclose(evaluator.params.iouThrs, threshold))[0])
    matched_gt, fps, duplicate_fps = set(), Counter(), 0
    for result in evaluator.evalImgs:
        if result is None or list(result['aRng']) != list(evaluator.params.areaRng[0]):
            continue
        eligible = {int(g) for g, ignored in zip(result['gtIds'], result['gtIgnore']) if not ignored}
        tp_gt = {int(g) for g, matched in zip(result['gtIds'], result['gtMatches'][index])
                 if matched and int(g) in eligible}
        matched_gt.update(tp_gt)
        unmatched = [evaluator.cocoDt.anns[int(d)] for d, m, ignored in zip(
            result['dtIds'], result['dtMatches'][index], result['dtIgnore'][index])
            if not m and not ignored]
        fps.update(fingerprint(x) for x in unmatched)
        if unmatched and tp_gt:
            overlaps = mask_util.iou(
                [rle_for_compute(x['segmentation']) for x in unmatched],
                [evaluator.cocoGt.annToRLE(evaluator.cocoGt.anns[g]) for g in sorted(tp_gt)],
                [0] * len(tp_gt))
            duplicate_fps += int(np.sum(np.max(overlaps, axis=1) >= threshold))
    return {'matched_gt': matched_gt, 'fp_geometries': fps,
            'tp': len(matched_gt), 'fp': sum(fps.values()), 'duplicate_fp': duplicate_fps}


def maximum_matching(edges, gt_count):
    owners = [-1] * gt_count
    def augment(prediction_index, visited):
        for gt_index in edges[prediction_index]:
            if gt_index in visited:
                continue
            visited.add(gt_index)
            if owners[gt_index] < 0 or augment(owners[gt_index], visited):
                owners[gt_index] = prediction_index
                return True
        return False
    return sum(augment(index, set()) for index in range(len(edges)))


def oracle_recall(dataset, left, right, threshold):
    gt_groups, predictions = defaultdict(list), defaultdict(list)
    from pycocotools.coco import COCO
    import contextlib
    with open(os.devnull, 'w') as sink, contextlib.redirect_stdout(sink):
        coco = COCO()
        coco.dataset = dataset
        coco.createIndex()
    for annotation in dataset['annotations']:
        if not annotation.get('iscrowd', 0):
            gt_groups[(int(annotation['image_id']), int(annotation['category_id']))].append(annotation)
    for item in left + right:
        predictions[(int(item['image_id']), int(item['category_id']))].append(item)
    matched, count = 0, 0
    for key, annotations in gt_groups.items():
        count += len(annotations)
        items = predictions[key]
        if not items:
            continue
        overlaps = mask_util.iou([rle_for_compute(x['segmentation']) for x in items],
                                [coco.annToRLE(x) for x in annotations], [0] * len(annotations))
        edges = [np.flatnonzero(row >= threshold).tolist() for row in overlaps]
        matched += maximum_matching(edges, len(annotations))
    return {'oracle_matched_gt': matched, 'gt_instances': count,
            'proposal_recall': float(matched) / count if count else None,
            'budget': 'union of two already-truncated top100 source files, at most 200/image',
            'not_an_ap_upper_bound': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ('gt-json', 'left-results', 'right-results', 'reference-results',
                  'controls-dir', 'output-dir'):
        parser.add_argument('--' + field, required=True)
    args = parser.parse_args()
    if os.path.exists(args.output_dir):
        raise RuntimeError('Refusing to overwrite output directory')
    def load(path):
        with open(path) as handle:
            return json.load(handle)
    dataset = load(args.gt_json)
    left, right, reference = [load(p) for p in (args.left_results, args.right_results, args.reference_results)]
    ref_eval = evaluate(dataset, reference)
    endpoint_summaries = {}
    for name, values in (('left', left), ('right', right)):
        ev = evaluate(dataset, values)
        endpoint_summaries[name] = {str(t): matching_summary(ev, t) for t in (0.5, 0.75)}
    report = {'protocol': vars(args), 'thresholds': {},
              'limitations': ['GT-assisted proposal recall is not deployment accuracy or AP upper bound.',
                              'TP identity follows each system COCO matching and can be reassigned.',
                              'FP geometry differences are multisets without score; ranking affects status.',
                              'Duplicate-FP means an unmatched same-category detection overlaps matched GT at threshold.']}
    for threshold in (0.5, 0.75):
        a = endpoint_summaries['left'][str(threshold)]['matched_gt']
        b = endpoint_summaries['right'][str(threshold)]['matched_gt']
        report['thresholds'][str(threshold)] = {
            'endpoint_gt_intersection': len(a & b), 'left_unique_gt': len(a - b),
            'right_unique_gt': len(b - a), 'endpoint_matching_union': len(a | b),
            'proposal_recall_oracle': oracle_recall(dataset, left, right, threshold), 'rules': {}}
    refs = {str(t): matching_summary(ref_eval, t) for t in (0.5, 0.75)}
    for rule in ('mask_nms', 'consensus', 'matched_no_penalty',
                 'linear_mask_soft_nms', 'gaussian_mask_soft_nms', 'matched_mask_union'):
        values = load(os.path.join(args.controls_dir, rule + '_coco_instances_results.json'))
        ev = evaluate(dataset, values)
        for threshold in (0.5, 0.75):
            current = matching_summary(ev, threshold)
            ref = refs[str(threshold)]
            row = {k: current[k] for k in ('tp', 'fp', 'duplicate_fp')}
            row.update({'reference_tp': ref['tp'], 'reference_fp': ref['fp'],
                        'reference_duplicate_fp': ref['duplicate_fp'],
                        'added_tp_gt': len(current['matched_gt'] - ref['matched_gt']),
                        'lost_tp_gt': len(ref['matched_gt'] - current['matched_gt']),
                        'retained_tp_gt': len(ref['matched_gt'] & current['matched_gt']),
                        'added_fp_geometry': sum((current['fp_geometries'] - ref['fp_geometries']).values()),
                        'removed_fp_geometry': sum((ref['fp_geometries'] - current['fp_geometries']).values())})
            report['thresholds'][str(threshold)]['rules'][rule] = row
        print('MECHANISM_RULE_COMPLETE ' + rule, flush=True)
    os.makedirs(args.output_dir)
    atomic_json_dump(report, os.path.join(args.output_dir, 'mechanisms_report.json'), indent=2)


if __name__ == '__main__':
    main()
