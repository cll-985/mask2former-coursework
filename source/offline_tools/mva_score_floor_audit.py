#!/usr/bin/env python3
"""Post-test exploratory audit of one frozen score floor and fixed score slice.

Uses saved predictions only. No rule selection, parameter sweep, or GPU use.
"""
import argparse
import datetime
import json
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from offline_tools.compare_fusion_rules import atomic_json_dump
from offline_tools.mva_full_coco_bootstrap import evaluate, sha256
from offline_tools.mva_fusion_mechanisms import matching_summary
from offline_tools.run_mva_phase2 import BASES, GT

PAIRS = ('seed42_seed3407', 'seed42_seed2026', 'seed3407_seed2026')
REFERENCE = {'seed42_seed3407': '3407', 'seed42_seed2026': '42',
             'seed3407_seed2026': '3407'}
RULES = ('concat_topk', 'mask_nms', 'matched_no_penalty', 'consensus',
         'linear_mask_soft_nms', 'gaussian_mask_soft_nms',
         'matched_mask_union', 'score_dominant_endpoint')


def metrics(evaluator):
    precision = evaluator.eval['precision']
    def mean(values):
        values = values[values > -1]
        return float(np.mean(values) * 100.0) if values.size else None
    thresholds = evaluator.params.iouThrs
    return {'AP': mean(precision[:, :, :, 0, -1]),
            'AP50': mean(precision[np.isclose(thresholds, 0.5), :, :, 0, -1]),
            'AP75': mean(precision[np.isclose(thresholds, 0.75), :, :, 0, -1]),
            'APs': mean(precision[:, :, :, 1, -1]),
            'APm': mean(precision[:, :, :, 2, -1]),
            'APl': mean(precision[:, :, :, 3, -1])}


def compare_records(report):
    rows = {}
    for pair in PAIRS:
        ref = report['systems'].get('baseline_seed' + REFERENCE[pair])
        if ref is None:
            continue
        rows[pair] = {}
        for rule in RULES:
            item = report['systems'].get(pair + '/' + rule)
            if item:
                rows[pair][rule] = {
                    'uniform_floor_ap': item['uniform_floor_metrics']['AP'],
                    'delta_vs_reference_uniform_floor_ap':
                    item['uniform_floor_metrics']['AP'] - ref['uniform_floor_metrics']['AP'],
                    'source_minus_uniform_floor_ap':
                    item['historical_raw_ap'] - item['uniform_floor_metrics']['AP']}
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase2-root', default='output_offline/MVA_phase2_20261015_v1')
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    os.chdir(str(ROOT))
    output = (ROOT / args.output_dir).resolve()
    output.relative_to(ROOT)
    output.mkdir(parents=True, exist_ok=False)
    phase2 = (ROOT / args.phase2_root).resolve()
    phase2.relative_to(ROOT)
    protocol_file = ROOT / 'refine-logs/MVA_SCORE_FLOOR_PROTOCOL_20261016.md'
    report = {'status': 'running', 'started_at': datetime.datetime.now(
        datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
        'protocol': {'floor': 0.001, 'diagnostic_score_slice': 0.5,
                     'pairs': list(PAIRS), 'rules': list(RULES),
                     'post_test_exploratory': True, 'test_split_access': False,
                     'gpu_disabled': os.environ.get('CUDA_VISIBLE_DEVICES') == '',
                     'protocol_sha256': sha256(str(protocol_file)),
                     'gt_sha256': sha256(GT)}, 'systems': {}}
    target = str(output / 'score_floor_report.json')
    atomic_json_dump(report, target, indent=2)
    started = time.monotonic()
    try:
        with open(GT) as handle:
            dataset = json.load(handle)
        sources = []
        for seed, path in BASES.items():
            with open(str(Path(path).parent.parent / 'metrics_summary.json')) as handle:
                old = json.load(handle)['metrics']['segm/AP']['last']
            sources.append(('baseline_seed' + seed, path, float(old)))
        for pair in PAIRS:
            with (phase2 / 'controls' / pair / 'controls_report.json').open() as handle:
                previous = json.load(handle)
            for rule in RULES:
                sources.append((pair + '/' + rule,
                    str(phase2 / 'controls' / pair / (rule + '_coco_instances_results.json')),
                    previous['rules'][rule]['evaluation']['segm']['AP']))
        for name, path, raw_ap in sources:
            with open(path) as handle:
                predictions = json.load(handle)
            filtered = [x for x in predictions if float(x['score']) >= 0.001]
            floor_evaluator = evaluate(dataset, filtered)
            floor_metrics = metrics(floor_evaluator)
            # Soft outputs already enforce the same floor: exact rerun check.
            if '_mask_soft_nms' in name and abs(floor_metrics['AP'] - raw_ap) >= 1e-6:
                raise AssertionError('Soft-NMS historical AP mismatch: ' + name)
            high = [x for x in predictions if float(x['score']) >= 0.5]
            high_evaluator = evaluate(dataset, high)
            counts = {}
            for threshold in (0.5, 0.75):
                summary = matching_summary(high_evaluator, threshold)
                tp, fp = summary['tp'], summary['fp']
                counts[str(threshold)] = {
                    'tp': tp, 'fp': fp, 'duplicate_fp': summary['duplicate_fp'],
                    'precision': float(tp) / (tp + fp) if tp + fp else None,
                    'matched_gt_ids': sorted(summary['matched_gt'])}
            report['systems'][name] = {
                'source': path, 'source_sha256': sha256(path),
                'historical_raw_ap': raw_ap, 'uniform_floor_metrics': floor_metrics,
                'raw_detections': len(predictions), 'uniform_floor_detections': len(filtered),
                'score_05_detections': len(high), 'score_05_counts': counts}
            report['completed'] = len(report['systems'])
            report['target'] = len(sources)
            report['elapsed_seconds'] = time.monotonic() - started
            report['comparisons'] = compare_records(report)
            atomic_json_dump(report, target, indent=2)
        report.update({'status': 'complete', 'elapsed_seconds': time.monotonic() - started})
        (output / 'exit_code.txt').write_text('0\n')
    except Exception:
        report.update({'status': 'failed', 'error': traceback.format_exc()})
        (output / 'exit_code.txt').write_text('1\n')
        raise
    finally:
        atomic_json_dump(report, target, indent=2)


if __name__ == '__main__':
    main()
