#!/usr/bin/env python3
"""Validate downloaded phase2 reports and export source-linked tables."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

BASE_AP = {'seed42_seed3407': 54.48288309947704,
           'seed42_seed2026': 53.663484698748775,
           'seed3407_seed2026': 54.48288309947704}
RULES = ('concat_topk', 'mask_nms', 'matched_no_penalty', 'consensus',
         'linear_mask_soft_nms', 'gaussian_mask_soft_nms',
         'matched_mask_union', 'score_dominant_endpoint')


def percentile(values, q):
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low = int(math.floor(position))
    high = int(math.ceil(position))
    return ordered[low] + (position - low) * (ordered[high] - ordered[low])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', required=True)
    args = parser.parse_args()
    folder = Path(args.input_dir)
    def load(name):
        with (folder / name).open(encoding='utf-8') as handle:
            return json.load(handle)
    state = load('status.json')
    if state['status'] != 'complete' or any(x.get('exit_code') != 0 for x in state['tasks'].values()):
        raise ValueError('Phase2 did not complete cleanly')
    boot = load('full_coco_bootstrap_report.json')
    with (folder / 'primary_resamples.csv').open(newline='') as handle:
        draws = list(csv.DictReader(handle))
    if len(draws) != 2000 or [int(x['draw']) for x in draws] != list(range(2000)):
        raise ValueError('Bootstrap count/order mismatch')
    values = [float(x['delta']) for x in draws]
    if any(not math.isfinite(x) for x in values):
        raise ValueError('Nonfinite bootstrap results')
    for row in draws:
        if abs(float(row['candidate_ap']) - float(row['baseline_ap']) - float(row['delta'])) > 1e-10:
            raise ValueError('CSV delta does not match AP difference')
    conditional = [float(x['delta']) for x in draws if int(x['complete_categories'])]
    ci = boot['complete_category_conditional_interval']
    expected = {'mean_delta': statistics.mean(conditional),
                'ci95_low': percentile(conditional, 0.025),
                'ci95_high': percentile(conditional, 0.975),
                'fraction_strictly_positive': sum(x > 0 for x in conditional) / len(conditional)}
    for key, value in expected.items():
        if abs(value - ci[key]) > 1e-10:
            raise ValueError('Independent CSV/report check failed: ' + key)
    flat, mechanisms, source_hashes = [], [], {}
    for pair, reference_ap in BASE_AP.items():
        d = load(pair + '_controls.json')
        if set(d['rules']) != set(RULES):
            raise ValueError('Missing or extra rules: ' + pair)
        for rule in RULES:
            item = d['rules'][rule]
            row = {'pair': pair, 'rule': rule, 'reference_ap': reference_ap,
                   'detections': item['output_detections']}
            row.update({k: item['evaluation']['segm'][k] for k in ('AP', 'AP50', 'AP75', 'APs', 'APm', 'APl')})
            row['delta_ap'] = row['AP'] - reference_ap
            row['relative_percent'] = 100 * row['delta_ap'] / reference_ap
            flat.append(row)
        if d['rules']['score_dominant_endpoint']['evaluation']['segm'] != d['rules']['consensus']['evaluation']['segm']:
            raise ValueError('Endpoint-degeneracy control differs from consensus')
        m = load(pair + '_mechanisms.json')
        for threshold, data in m['thresholds'].items():
            for rule, counts in data['rules'].items():
                if counts['tp'] != counts['reference_tp'] + counts['added_tp_gt'] - counts['lost_tp_gt']:
                    raise ValueError('TP decomposition invariant failed')
                row = {'pair': pair, 'iou': threshold, 'rule': rule}
                row.update(counts)
                mechanisms.append(row)
    for path in sorted(folder.glob('*.json')):
        if path.name != 'validated_summary.json':
            source_hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    for name, rows in (('controls_table.csv', flat), ('mechanisms_table.csv', mechanisms)):
        with (folder / name).open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    result = {'status': 'validated', 'source_hashes': source_hashes,
              'bootstrap_csv_sha256': hashlib.sha256((folder / 'primary_resamples.csv').read_bytes()).hexdigest(),
              'bootstrap_rows': len(draws), 'complete_category_rows': len(conditional),
              'independently_recomputed_interval': expected,
              'control_rows': len(flat), 'mechanism_rows': len(mechanisms),
              'controls': flat, 'pair_statistics_are_descriptive_not_independent': True}
    with (folder / 'validated_summary.json').open('w', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
        handle.write('\n')
    print(json.dumps({k: result[k] for k in ('status', 'bootstrap_rows', 'complete_category_rows',
                                           'control_rows', 'mechanism_rows',
                                           'independently_recomputed_interval')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
