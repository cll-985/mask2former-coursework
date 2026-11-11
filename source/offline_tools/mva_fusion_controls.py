#!/usr/bin/env python3
"""Frozen post-test exploratory controls; no training or GPU imports."""
import argparse
import copy
import json
import math
import os
import sys
import time
from collections import Counter

import numpy as np
from pycocotools import mask as mask_util

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from offline_tools.compare_fusion_rules import (
    atomic_json_dump, concat_topk, group_results, mask_nms,
    matched_no_penalty, prune_topk)
from offline_tools.fuse_coco_instances import (
    evaluate_results, fuse_results, greedy_matches, rle_for_compute)

RULES = ('concat_topk', 'mask_nms', 'matched_no_penalty', 'consensus',
         'linear_mask_soft_nms', 'gaussian_mask_soft_nms',
         'matched_mask_union', 'score_dominant_endpoint')


def soft_nms(base, other, kind, threshold=0.5, sigma=0.5,
             score_floor=0.001, max_dets=100):
    output, stats = [], Counter()
    for key, values in sorted(group_results(base + other).items()):
        values = sorted(values, key=lambda x: -float(x['score']))
        rles = [rle_for_compute(x['segmentation']) for x in values]
        ious = mask_util.iou(rles, rles, [0] * len(rles))
        scores = np.asarray([x['score'] for x in values], dtype=np.float64)
        active = list(range(len(values)))
        while active:
            # Stable tie order retains the original sorted candidate order.
            winner = max(active, key=lambda i: scores[i])
            active.remove(winner)
            if scores[winner] < score_floor:
                stats['removed_below_floor'] += 1 + len(active)
                break
            item = copy.deepcopy(values[winner])
            item['score'] = float(scores[winner])
            output.append(item)
            for index in active:
                overlap = float(ious[winner, index])
                if kind == 'linear':
                    factor = 1.0 - overlap if overlap >= threshold else 1.0
                else:
                    factor = math.exp(-(overlap * overlap) / sigma)
                scores[index] *= factor
        stats['groups'] += 1
    stats['kept_before_topk'] = len(output)
    return prune_topk(output, max_dets), dict(stats)


def binary_geometry_control(base, other, kind, threshold=0.5, max_dets=100):
    left_groups, right_groups = group_results(base), group_results(other)
    output, stats = [], Counter()
    for key in sorted(set(left_groups) | set(right_groups)):
        left = sorted(left_groups[key], key=lambda x: -float(x['score']))
        right = sorted(right_groups[key], key=lambda x: -float(x['score']))
        matches = greedy_matches(left, right, threshold)
        used_left, used_right = set(), set()
        for i, j, _ in matches:
            a, b = left[i], right[j]
            used_left.add(i)
            used_right.add(j)
            item = copy.deepcopy(a if a['score'] >= b['score'] else b)
            endpoint = rle_for_compute(item['segmentation'])
            use_union = kind == 'union' or float(a['score']) == float(b['score'])
            if use_union:
                merged = mask_util.merge([rle_for_compute(a['segmentation']),
                                          rle_for_compute(b['segmentation'])])
                stats['union_equals_selected_endpoint'] += int(
                    float(mask_util.iou([merged], [endpoint], [0])[0, 0]) == 1.0)
                item['bbox'] = mask_util.toBbox(merged).tolist()
                if isinstance(merged.get('counts'), bytes):
                    merged['counts'] = merged['counts'].decode('utf-8')
                item['segmentation'] = merged
                stats['union_applied'] += 1
            else:
                stats['geometry_equals_selected_endpoint'] += 1
            item['score'] = (float(a['score']) + float(b['score'])) / 2.0
            output.append(item)
            stats['matched'] += 1
        for values, used in ((left, used_left), (right, used_right)):
            for index, source in enumerate(values):
                if index not in used:
                    item = copy.deepcopy(source)
                    item['score'] = float(item['score']) / 2.0
                    output.append(item)
                    stats['unmatched'] += 1
    stats['not_probability_mask_fusion'] = True
    return prune_topk(output, max_dets), dict(stats)


def apply_rule(rule, base, other):
    if rule == 'concat_topk':
        return concat_topk(base, other, 100)
    if rule == 'mask_nms':
        return mask_nms(base, other, 0.5, 100)
    if rule == 'matched_no_penalty':
        return matched_no_penalty(base, other, 0.5, 100)
    if rule == 'consensus':
        return fuse_results(base, other, 0.5, 100)
    if rule in ('linear_mask_soft_nms', 'gaussian_mask_soft_nms'):
        return soft_nms(base, other, rule.split('_')[0])
    return binary_geometry_control(
        base, other, 'union' if rule == 'matched_mask_union' else 'endpoint')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ('base-results', 'other-results', 'gt-json', 'output-dir'):
        parser.add_argument('--' + field, required=True)
    parser.add_argument('--expected-mask-nms-ap', type=float)
    parser.add_argument('--expected-consensus-ap', type=float)
    args = parser.parse_args()
    if os.path.exists(args.output_dir):
        raise RuntimeError('Output directory exists; refusing to overwrite')
    with open(args.base_results) as handle:
        base = json.load(handle)
    with open(args.other_results) as handle:
        other = json.load(handle)
    with open(args.gt_json) as handle:
        gt = json.load(handle)
    allowed_images = {int(x['id']) for x in gt['images']}
    allowed_categories = {int(x['id']) for x in gt['categories']}
    for item in base + other:
        if int(item['image_id']) not in allowed_images or int(item['category_id']) not in allowed_categories:
            raise ValueError('Prediction outside frozen GT IDs/categories')
        if not math.isfinite(float(item['score'])) or float(item['score']) < 0:
            raise ValueError('Invalid score')
    os.makedirs(args.output_dir)
    report = {'protocol': {'rules': list(RULES), 'mask_iou': 0.5,
                          'max_dets_per_image': 100, 'soft_score_floor': 0.001,
                          'gaussian_sigma': 0.5, 'post_test_exploratory': True,
                          'raw_mask_logits_available': False}, 'rules': {}}
    for rule in RULES:
        started = time.monotonic()
        output, stats = apply_rule(rule, base, other)
        path = os.path.join(args.output_dir, rule + '_coco_instances_results.json')
        atomic_json_dump(output, path)
        stats['evaluation'] = evaluate_results(args.gt_json, path)
        stats['seconds_including_evaluation'] = time.monotonic() - started
        stats['output_detections'] = len(output)
        report['rules'][rule] = stats
        if rule == 'mask_nms' and args.expected_mask_nms_ap is not None:
            actual = stats['evaluation']['segm']['AP']
            if abs(actual - args.expected_mask_nms_ap) >= 1e-6:
                raise AssertionError('Historical AP mismatch: {} != {}'.format(
                    actual, args.expected_mask_nms_ap))
        if rule == 'consensus' and args.expected_consensus_ap is not None:
            actual = stats['evaluation']['segm']['AP']
            if abs(actual - args.expected_consensus_ap) >= 1e-6:
                raise AssertionError('Historical consensus AP mismatch: {} != {}'.format(
                    actual, args.expected_consensus_ap))
        atomic_json_dump(report, os.path.join(args.output_dir, 'controls_report.json'), indent=2)
        print('RULE_COMPLETE {} AP={:.8f} seconds={:.2f}'.format(
            rule, stats['evaluation']['segm']['AP'], stats['seconds_including_evaluation']), flush=True)


if __name__ == '__main__':
    main()
