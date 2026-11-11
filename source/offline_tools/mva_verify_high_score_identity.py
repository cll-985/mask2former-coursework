#!/usr/bin/env python3
"""Exact saved-prediction multiset comparison at fixed score >= 0.5."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase2-root', default='output_offline/MVA_phase2_20261015_v1')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    phase2 = (root / args.phase2_root).resolve()
    phase2.relative_to(root)
    target = (root / args.output).resolve()
    target.relative_to(root)
    report = {'threshold': 0.5, 'comparison': 'exact multiset of image ID, class, mask RLE and score',
              'pairs': {}, 'post_test_exploratory': True}
    for pair in ('seed42_seed3407', 'seed42_seed2026', 'seed3407_seed2026'):
        counters, hashes = {}, {}
        for rule in ('mask_nms', 'linear_mask_soft_nms'):
            path = phase2 / 'controls' / pair / (rule + '_coco_instances_results.json')
            raw = path.read_bytes()
            hashes[rule] = hashlib.sha256(raw).hexdigest()
            predictions = json.loads(raw.decode('utf-8'))
            counters[rule] = Counter((int(x['image_id']), int(x['category_id']),
                float(x['score']), json.dumps(x['segmentation'], sort_keys=True))
                for x in predictions if float(x['score']) >= 0.5)
        hard, soft = counters['mask_nms'], counters['linear_mask_soft_nms']
        report['pairs'][pair] = {'identical': hard == soft,
            'hard_count': sum(hard.values()), 'linear_count': sum(soft.values()),
            'hard_only': sum((hard - soft).values()),
            'linear_only': sum((soft - hard).values()), 'source_sha256': hashes}
    with target.open('x', encoding='utf-8') as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
    print(json.dumps(report, sort_keys=True))


if __name__ == '__main__':
    main()
