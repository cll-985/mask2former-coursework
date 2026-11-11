"""Compare fixed damage segmenters at matched empirical recall."""

import argparse
import contextlib
import copy
import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np


SYSTEMS = ('single', 'hard', 'linear', 'gaussian')
TARGETS = {'all': (10, 15, 20, 25), 'Glass_crack': (5, 10, 15)}
IOU = (.50, .75)
SEED = 20261108
DRAWS = 2000
FLOOR = .001


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def native_matches(dataset, predictions):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    with open(os.devnull, 'w') as sink, contextlib.redirect_stdout(sink):
        gt = COCO()
        gt.dataset = copy.deepcopy(dataset)
        gt.createIndex()
        if predictions:
            dt = gt.loadRes(predictions)
        else:
            dt = COCO()
            dt.dataset = dict(images=gt.dataset['images'], categories=gt.dataset['categories'],
                              annotations=[])
            dt.createIndex()
        ev = COCOeval(gt, dt, 'segm')
        ev.params.areaRng = [ev.params.areaRng[0]]
        ev.params.areaRngLbl = ['all']
        ev.params.iouThrs = np.array(IOU)
        ev.params.maxDets = [100]
        ev.evaluate()
    return ev.evalImgs


def prediction_events(matches, image_ids, category_id, threshold):
    image_index = {image_id: i for i, image_id in enumerate(image_ids)}
    gt = np.zeros(len(image_ids), dtype=np.int32)
    tp_scores, tp_images = [], []
    fp_scores = [[] for _ in image_ids]
    for row in matches:
        if row is None or (category_id is not None and row['category_id'] != category_id):
            continue
        i = image_index[row['image_id']]
        gt[i] += np.count_nonzero(np.logical_not(row['gtIgnore']))
        scores = np.asarray(row['dtScores'])
        valid = np.logical_not(row['dtIgnore'][threshold])
        is_match = row['dtMatches'][threshold] > 0
        tp = scores[valid & is_match]
        tp_scores.extend(tp)
        tp_images.extend([i] * len(tp))
        fp_scores[i].extend(scores[valid & ~is_match])
    scores = np.asarray(tp_scores, dtype=np.float64)
    order = np.argsort(-scores, kind='stable')
    return (gt, scores[order], np.asarray(tp_images, dtype=np.int32)[order],
            [np.sort(scores_i) for scores_i in fp_scores])


def operating_points(events, weights, targets):
    """Select a score group using only TP events, then count FP above its cutoff."""
    gt, tp_scores, tp_images, fp_scores = events
    totals = weights @ gt
    required = (totals[:, None] * np.asarray(targets) + 99) // 100
    chosen = np.full(required.shape, np.nan)
    true_counts = np.zeros(required.shape, dtype=np.int32)
    maximum = np.zeros(len(weights), dtype=np.int32)
    if tp_scores.size:
        for start in range(0, len(weights), 64):
            block = slice(start, start + 64)
            cumulative = np.cumsum(weights[block, tp_images], axis=1, dtype=np.int32)
            maximum[block] = cumulative[:, -1]
            for col in range(len(targets)):
                eligible = (required[block, col] > 0) & (cumulative[:, -1] >= required[block, col])
                rows = np.flatnonzero(eligible)
                crossings = (cumulative[rows] >= required[start + rows, col, None]).argmax(axis=1)
                cutoff = tp_scores[crossings]
                ends = np.searchsorted(-tp_scores, -cutoff, side='right') - 1
                chosen[start + rows, col] = cutoff
                true_counts[start + rows, col] = cumulative[rows, ends]

    cutoffs = np.unique(chosen[np.isfinite(chosen)])
    false_counts = np.full(chosen.shape, np.nan)
    if cutoffs.size:
        fp_at_cutoff = np.asarray([len(s) - np.searchsorted(s, cutoffs, side='left')
                                   for s in fp_scores], dtype=np.int32)
        for col in range(len(targets)):
            rows = np.flatnonzero(np.isfinite(chosen[:, col]))
            columns = np.searchsorted(cutoffs, chosen[rows, col])
            for start in range(0, len(rows), 64):
                r, c = rows[start:start + 64], columns[start:start + 64]
                false_counts[r, col] = np.einsum('ij,ij->i', weights[r], fp_at_cutoff[:, c].T)
    return chosen, true_counts, false_counts, maximum, totals


def self_test():
    from pycocotools import mask as mask_util

    def mask(top, left):
        data = np.zeros((8, 8), dtype=np.uint8, order='F')
        data[top:top + 3, left:left + 3] = 1
        return mask_util.encode(data)

    dataset = dict(images=[dict(id=i, width=8, height=8) for i in (3, 8, 9, 11)],
                   categories=[dict(id=1, name='Glass_crack')], annotations=[
        dict(id=1, image_id=3, category_id=1, segmentation=mask(0, 0), area=9, iscrowd=0),
        dict(id=2, image_id=8, category_id=1, segmentation=mask(0, 0), area=9, iscrowd=1),
        dict(id=3, image_id=9, category_id=1, segmentation=mask(0, 0), area=9, iscrowd=0),
        dict(id=4, image_id=11, category_id=1, segmentation=mask(0, 0), area=9, iscrowd=0),
    ])
    preds = [dict(image_id=i, category_id=1, score=s, segmentation=mask(t, t))
             for i, s, t in [(3, .5, 0), (3, .5, 5), (8, .5, 0), (8, .5, 5), (9, .4, 0)]]
    image_ids = [3, 8, 9, 11]
    weights = np.array([[1, 1, 1, 1], [2, 1, 0, 1], [0, 4, 0, 0]], dtype=np.int32)
    events = prediction_events(native_matches(dataset, copy.deepcopy(preds)), image_ids, None, 1)
    score, tp, fp, maximum, total = operating_points(events, weights, (30, 60, 90))
    assert score[0, 0] == .5 and score[0, 1] == .4
    assert (tp[0, 0], fp[0, 0]) == (1, 2)
    assert (tp[1, 0], fp[1, 0]) == (2, 3)
    assert np.isnan(score[0, 2]) and np.isnan(score[2]).all()
    assert list(maximum) == [2, 2, 0] and list(total) == [3, 3, 0]
    for threshold in (.5, .4):
        retained = [dict(x) for x in preds if x['score'] >= threshold]
        direct = prediction_events(native_matches(dataset, retained), image_ids, None, 1)
        for row, col in zip(*np.where(score == threshold)):
            assert weights[row, direct[2]].sum() == tp[row, col]
            assert sum(w * len(s) for w, s in zip(weights[row], direct[3])) == fp[row, col]
    print('COCO tie, crowd, repeated-image and infeasibility witness passed', flush=True)


def load_predictions(path, image_ids, category_ids):
    result = []
    with Path(path).open() as handle:
        loaded = json.load(handle)
    image_set = set(image_ids)
    counts = {}
    for item in loaded:
        score = float(item['score'])
        if not np.isfinite(score) or item['image_id'] not in image_set or item['category_id'] not in category_ids:
            raise ValueError('Invalid score, image or category in prediction')
        if score >= FLOOR:
            item.pop('bbox', None)
            item.pop('area', None)
            item['score'] = score
            image_id = item['image_id']
            counts[image_id] = counts.get(image_id, 0) + 1
            result.append(item)
    if max(counts.values(), default=0) > 100:
        raise ValueError('Archived predictions exceed the 100-per-image budget')
    return result


def write_csv(path, rows):
    with path.open('x', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(manifest_path, output):
    with manifest_path.open(encoding='utf-8') as handle:
        manifest = json.load(handle)
    inputs = manifest['inputs']
    hashes = {name: sha256(entry['path']) for name, entry in inputs.items()}
    for name, actual in hashes.items():
        if inputs[name].get('sha256') != actual:
            raise ValueError('Input SHA-256 mismatch for ' + name)
    with Path(inputs['gt']['path']).open() as handle:
        dataset = json.load(handle)
    ids = [annotation['id'] for annotation in dataset['annotations']]
    if len(ids) != len(set(ids)) or min(ids) < 0:
        raise ValueError('Expected unique nonnegative GT IDs')
    offset = int(0 in ids)
    if offset:
        for annotation in dataset['annotations']:
            annotation['id'] += offset
    image_ids = sorted(x['id'] for x in dataset['images'])
    if len(image_ids) != manifest['images'] or len(set(image_ids)) != len(image_ids):
        raise ValueError('Unexpected cohort image roster')
    categories = {x['name']: x['id'] for x in dataset['categories']}
    if len(categories) != 9 or 'Glass_crack' not in categories:
        raise ValueError('Unexpected category roster')

    rng = np.random.RandomState(SEED)
    n = len(image_ids)
    weights = np.ones((DRAWS + 1, n), dtype=np.int32)
    for row in weights[1:]:
        row[:] = np.bincount(rng.randint(0, n, size=n), minlength=n)
    results = {}
    expected = manifest.get('archived_max_tp_iou75', {})
    for name in SYSTEMS:
        print('Matching ' + name, flush=True)
        predictions = load_predictions(inputs[name]['path'], image_ids, set(categories.values()))
        matches = native_matches(dataset, predictions)
        del predictions
        for group, targets in TARGETS.items():
            for i, iou in enumerate(IOU):
                events = prediction_events(matches, image_ids, categories.get(group), i)
                results[name, group, iou] = operating_points(events, weights, targets)
        print('Finished ' + name, flush=True)
        del matches

    points, contrasts, draws = [], [], []
    witness = {}
    for name in SYSTEMS:
        maximum = int(results[name, 'all', .75][3][0])
        witness[name] = dict(replayed_max_tp=maximum, archived_max_tp=expected.get(name))
        if name in expected and maximum != expected[name]:
            raise ValueError('Full-budget IoU75 match count mismatch for ' + name)
    for group, targets in TARGETS.items():
        for iou in IOU:
            reference_totals = results['single', group, iou][4]
            for col, target in enumerate(targets):
                tag = dict(group=group, mask_iou=iou, recall_target=target / 100.0)
                for name in SYSTEMS:
                    score, tp, fp, maximum, total = results[name, group, iou]
                    if not np.array_equal(total, reference_totals):
                        raise ValueError('Eligible GT changed across systems')
                    feasible = bool(np.isfinite(score[0, col]))
                    points.append(dict(**tag, system=name, reached=feasible,
                                       gt=int(total[0]), maximum_recall=float(maximum[0] / total[0]),
                                       threshold=float(score[0, col]) if feasible else None,
                                       achieved_recall=float(tp[0, col] / total[0]) if feasible else None,
                                       tp=int(tp[0, col]) if feasible else None,
                                       fp=int(fp[0, col]) if feasible else None,
                                       fp_per_image=float(fp[0, col] / n) if feasible else None,
                                       bootstrap_feasible=int(np.isfinite(score[1:, col]).sum())))
                for candidate, reference in [('hard', 'single'), ('linear', 'single'),
                                             ('gaussian', 'single'), ('linear', 'hard')]:
                    delta = (results[candidate, group, iou][2][:, col] -
                             results[reference, group, iou][2][:, col]) / n
                    finite = delta[1:][np.isfinite(delta[1:])]
                    interval = np.percentile(finite, (2.5, 97.5)).tolist() if len(finite) >= 100 else [None, None]
                    contrast = dict(**tag, candidate=candidate, reference=reference)
                    contrasts.append(dict(**contrast, delta_fp_per_image=float(delta[0]) if np.isfinite(delta[0]) else None,
                                          ci95_low=interval[0], ci95_high=interval[1],
                                          feasible_draws=len(finite), total_draws=DRAWS,
                                          interval_conditional=0 < len(finite) < DRAWS))
                    draws.extend(dict(**contrast, draw=i, delta_fp_per_image=float(x) if np.isfinite(x) else None)
                                 for i, x in enumerate(delta[1:], 1))
    for name, entries in (('points.csv', points), ('contrasts.csv', contrasts), ('draws.csv', draws)):
        write_csv(output / name, entries)
    report = dict(status='complete', exploratory=True, cohort=manifest['cohort'], images=n,
                  single_seed=manifest['single_seed'], rng_seed=SEED, draws=DRAWS, floor=FLOOR,
                  gt_id_offset=offset, input_sha256=hashes, source_sha256=sha256(__file__),
                  manifest_sha256=sha256(manifest_path), native_max_tp_iou75=witness)
    with (output / 'report.json').open('x', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2)
    print('Completed all operating points and paired intervals', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    self_test()
    if not args.self_test:
        if args.manifest is None or args.output is None:
            parser.error('--manifest and --output are required')
        args.output.mkdir()
        run(args.manifest, args.output)


if __name__ == '__main__':
    main()
