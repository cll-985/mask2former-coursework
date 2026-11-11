"""Retrospective cross-fitted thresholds at fixed calibration FP budgets."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from mva_operating_points import (
    FLOOR, SYSTEMS, IOU, load_predictions, native_matches, prediction_events, sha256)


BUDGETS = (.5, 1., 2.)
FOLDS = 5
DRAWS = 2000
SEED = 20261109
FOLD_SALT = 'mva-fp-crossfit-20261108:'
CONTRASTS = (('hard', 'single'), ('linear', 'single'),
             ('gaussian', 'single'), ('linear', 'hard'))


def fold_ids(image_ids):
    order = sorted(range(len(image_ids)), key=lambda i: (
        hashlib.sha256((FOLD_SALT + str(image_ids[i])).encode()).hexdigest(), image_ids[i]))
    folds = np.empty(len(image_ids), dtype=np.int32)
    for rank, i in enumerate(order):
        folds[i] = rank % FOLDS
    return folds


def draw_weights(folds):
    rng = np.random.RandomState(SEED)
    weights = np.ones((DRAWS + 1, len(folds)), dtype=np.int16)
    for row in weights[1:]:
        row.fill(0)
        for fold in range(FOLDS):
            positions = np.flatnonzero(folds == fold)
            row[positions] = np.bincount(rng.randint(0, len(positions), size=len(positions)),
                                        minlength=len(positions))
    return weights


def fp_events(events):
    by_image = events[3]
    sizes = [len(values) for values in by_image]
    scores = np.concatenate(by_image)
    images = np.repeat(np.arange(len(by_image), dtype=np.int32), sizes)
    order = np.argsort(-scores, kind='stable')
    return scores[order], images[order]


def select_cutoffs(scores, images, weights, calibration_size):
    """Return lowest inclusive cutoffs satisfying complete FP score-group budgets."""
    caps = np.floor(np.asarray(BUDGETS) * calibration_size).astype(np.int64)
    cutoffs = np.full((len(weights), len(caps)), FLOOR, dtype=np.float64)
    if not len(scores):
        return cutoffs
    for start in range(0, len(weights), 32):
        block = weights[start:start + 32]
        prefix = min(len(scores), int(caps[-1]) * 2 + 64)
        while True:
            cumulative = np.cumsum(block[:, images[:prefix]], axis=1, dtype=np.int32)
            if prefix == len(scores) or np.all(cumulative[:, -1] > caps[-1]):
                break
            prefix = min(len(scores), prefix * 2)
        for col, cap in enumerate(caps):
            crossed = cumulative[:, -1] > cap
            rows = np.flatnonzero(crossed)
            indices = (cumulative[rows] > cap).argmax(axis=1)
            cutoffs[start + rows, col] = np.nextafter(scores[indices], np.inf)
    return cutoffs


def weighted_counts(scores, images, weights, cutoffs):
    counts = np.zeros(cutoffs.shape, dtype=np.int64)
    for col in range(cutoffs.shape[1]):
        eligible = scores >= cutoffs[:, col].min()
        selected_scores, selected_images = scores[eligible], images[eligible]
        for start in range(0, len(weights), 64):
            retained = selected_scores[None, :] >= cutoffs[start:start + 64, col, None]
            counts[start:start + 64, col] = np.sum(
                weights[start:start + 64, selected_images] * retained, axis=1, dtype=np.int64)
    return counts


def counts_by_image(events, thresholds, holdout, image_count):
    gt, tp_scores, tp_images, fp_by_image = events
    tp = np.bincount(tp_images[holdout[tp_images] &
                              (tp_scores >= thresholds)], minlength=image_count)
    fp = np.asarray([np.count_nonzero(values >= thresholds) if holdout[i] else 0
                     for i, values in enumerate(fp_by_image)], dtype=np.int64)
    return tp, fp


def write_csv(path, rows):
    with path.open('x', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def pure_witness():
    rng = np.random.RandomState(719)
    for _ in range(80):
        size = int(rng.randint(1, 12))
        scores = np.sort(rng.choice([.001, .1, .3, .5, .9], size=rng.randint(80)))[::-1]
        images = rng.randint(0, size, size=len(scores))
        weights = np.asarray([np.bincount(rng.randint(0, size, size=size), minlength=size)
                              for _ in range(7)], dtype=np.int16)
        actual = select_cutoffs(scores, images, weights, size)
        candidates = sorted(set([FLOOR] + [float(np.nextafter(s, np.inf)) for s in scores]))
        for row, weight in enumerate(weights):
            for col, budget in enumerate(BUDGETS):
                cap = int(np.floor(budget * size))
                expected = next(t for t in candidates if weight[images[scores >= t]].sum() <= cap)
                assert actual[row, col] == expected
                assert weight[images[scores >= actual[row, col]]].sum() <= cap
    ids = list(range(37))
    folds = fold_ids(ids)
    assert max(np.bincount(folds)) - min(np.bincount(folds)) <= 1
    assert np.array_equal(folds, fold_ids(ids))
    print('Weighted complete-tie FP-budget selection witness passed', flush=True)


def run(manifest_path, output, *, systems=SYSTEMS, comparisons=CONTRASTS):
    manifest = json.loads(manifest_path.read_text())
    for name, entry in manifest['inputs'].items():
        if sha256(entry['path']) != entry['sha256']:
            raise ValueError('Input hash mismatch for ' + name)
    dataset = json.loads(Path(manifest['inputs']['gt']['path']).read_text())
    annotation_ids = [annotation['id'] for annotation in dataset['annotations']]
    if len(set(annotation_ids)) != len(annotation_ids) or min(annotation_ids) < 0:
        raise ValueError('Invalid annotation IDs')
    offset = int(0 in annotation_ids)
    for annotation in dataset['annotations']:
        annotation['id'] += offset
    image_ids = sorted(image['id'] for image in dataset['images'])
    if len(set(image_ids)) != len(image_ids) or len(image_ids) != manifest['images']:
        raise ValueError('Unexpected image roster')
    category_ids = sorted(category['id'] for category in dataset['categories'])
    names = {category['id']: category['name'] for category in dataset['categories']}
    expected_categories = manifest.get('category_names')
    if expected_categories is None:
        if len(category_ids) != 9 or 'Glass_crack' not in names.values():
            raise ValueError('Unexpected category roster')
    elif sorted(names.values()) != sorted(expected_categories):
        raise ValueError('Category names differ from the pinned manifest')
    n, categories = len(image_ids), len(category_ids)
    folds = fold_ids(image_ids)
    weights = draw_weights(folds)
    write_csv(output / 'folds.csv', [dict(image_id=image, fold=int(fold))
                                    for image, fold in zip(image_ids, folds)])
    boot_counts = {}
    calibration_rows, points = [], []
    gt_counts = None
    native_counts = {}

    for system in systems:
        print('Matching ' + system, flush=True)
        predictions = load_predictions(manifest['inputs'][system]['path'], image_ids, set(category_ids))
        matches = native_matches(dataset, predictions)
        del predictions
        events_by_group = {
            (i, cat): prediction_events(matches, image_ids, cat, i)
            for i in range(len(IOU)) for cat in category_ids}
        all_events = prediction_events(matches, image_ids, None, 1)
        del matches
        this_gt = np.asarray([events_by_group[1, cat][0] for cat in category_ids])
        if gt_counts is None:
            gt_counts = this_gt
        else:
            assert np.array_equal(gt_counts, this_gt)
        native_counts[system] = len(all_events[1])
        expected = manifest.get('archived_max_tp_iou75', {}).get(system)
        if expected is not None and expected != native_counts[system]:
            raise ValueError('Archived full-budget TP mismatch for ' + system)
        scores, images = fp_events(all_events)
        counts = np.zeros((len(BUDGETS), len(IOU), categories, n, 2), dtype=np.int32)
        boot = np.zeros((DRAWS + 1, len(BUDGETS), 2), dtype=np.int64)
        thresholds = []
        for fold in range(FOLDS):
            heldout = folds == fold
            train = ~heldout
            cal_size = int(train.sum())
            keep = train[images]
            cutoffs = select_cutoffs(scores[keep], images[keep], weights, cal_size)
            thresholds.append(cutoffs)
            tp_keep = heldout[all_events[2]]
            fp_keep = heldout[images]
            boot[:, :, 0] += weighted_counts(all_events[1][tp_keep], all_events[2][tp_keep],
                                             weights, cutoffs)
            boot[:, :, 1] += weighted_counts(scores[fp_keep], images[fp_keep], weights, cutoffs)
            for b, budget in enumerate(BUDGETS):
                cutoff = cutoffs[0, b]
                cal_fp = int(np.count_nonzero(train[images] & (scores >= cutoff)))
                assert cal_fp <= int(np.floor(budget * cal_size))
                for i in range(len(IOU)):
                    for c, cat in enumerate(category_ids):
                        tp, fp = counts_by_image(events_by_group[i, cat], cutoff, heldout, n)
                        counts[b, i, c, :, 0] += tp
                        counts[b, i, c, :, 1] += fp
                calibration_rows.append(dict(system=system, fold=fold, budget=budget,
                                             calibration_images=cal_size, heldout_images=int(heldout.sum()),
                                             cutoff=cutoff, calibration_fp=cal_fp,
                                             calibration_fp_per_image=cal_fp / cal_size,
                                             heldout_tp=int(counts[b, 1, :, heldout, 0].sum()),
                                             heldout_fp=int(counts[b, 1, :, heldout, 1].sum())))
            print(system, 'fold', fold, 'complete', flush=True)
        for b, budget in enumerate(BUDGETS):
            total_tp = int(counts[b, 1, :, :, 0].sum())
            total_fp = int(counts[b, 1, :, :, 1].sum())
            assert (total_tp, total_fp) == tuple(boot[0, b])
            for i, iou in enumerate(IOU):
                groups = [('all', list(range(categories)))] + [(names[cat], [c])
                                                               for c, cat in enumerate(category_ids)]
                for group, selected in groups:
                    gt = int(gt_counts[selected].sum())
                    tp = int(counts[b, i, selected, :, 0].sum())
                    fp = int(counts[b, i, selected, :, 1].sum())
                    per_class_gt = gt_counts[selected].sum(axis=1)
                    per_class_tp = counts[b, i, selected, :, 0].sum(axis=1)
                    valid = per_class_gt > 0
                    points.append(dict(system=system, calibration_budget=budget, mask_iou=iou,
                                       group=group, images=n, gt=gt, tp=tp, fp=fp,
                                       recall_percent=100 * tp / gt if gt else None,
                                       macro_recall_percent=float(np.mean(100 * per_class_tp[valid] /
                                                                          per_class_gt[valid])),
                                       fp_per_image=fp / n))
        boot_counts[system] = boot
        np.savez_compressed(output / (system + '_image_counts.npz'), counts=counts, gt=gt_counts,
                            folds=folds, image_ids=image_ids, category_ids=category_ids,
                            bootstrap_counts=boot, bootstrap_cutoffs=np.asarray(thresholds))
        print('Completed ' + system, flush=True)

    bootstrap_gt = weights @ gt_counts.sum(axis=0)
    contrasts = []
    for b, budget in enumerate(BUDGETS):
        for candidate, reference in comparisons:
            delta_recall = 100 * (boot_counts[candidate][:, b, 0] -
                                  boot_counts[reference][:, b, 0]) / bootstrap_gt
            delta_fp = (boot_counts[candidate][:, b, 1] -
                        boot_counts[reference][:, b, 1]) / n
            rlo, rhi = np.percentile(delta_recall[1:], (2.5, 97.5))
            flo, fhi = np.percentile(delta_fp[1:], (2.5, 97.5))
            contrasts.append(dict(calibration_budget=budget, candidate=candidate, reference=reference,
                                  delta_recall_points=float(delta_recall[0]),
                                  recall_ci95_low=rlo, recall_ci95_high=rhi,
                                  delta_fp_per_image=float(delta_fp[0]),
                                  fp_ci95_low=flo, fp_ci95_high=fhi, draws=DRAWS))
    draw_rows = []
    for d in range(1, DRAWS + 1):
        for b, budget in enumerate(BUDGETS):
            row = dict(draw=d, calibration_budget=budget, gt=int(bootstrap_gt[d]), images=n)
            for system in systems:
                row[system + '_tp'] = int(boot_counts[system][d, b, 0])
                row[system + '_fp'] = int(boot_counts[system][d, b, 1])
            draw_rows.append(row)
    write_csv(output / 'points.csv', points)
    write_csv(output / 'contrasts.csv', contrasts)
    write_csv(output / 'calibration.csv', calibration_rows)
    write_csv(output / 'draws.csv', draw_rows)
    for name, entry in manifest['inputs'].items():
        if sha256(entry['path']) != entry['sha256']:
            raise ValueError('End-of-run input hash mismatch for ' + name)
    report = dict(status='complete', exploratory=True, cohort=manifest['cohort'], images=n,
                  folds=FOLDS, budgets=BUDGETS, draws=DRAWS, seed=SEED, gt_id_offset=offset,
                  same_cutoff_for_all_iou_and_classes=True, threshold_refit_per_draw=True,
                  source_sha256=sha256(__file__), reference_source_sha256=sha256(
                      Path(__file__).with_name('mva_operating_points.py')),
                  manifest_sha256=sha256(manifest_path),
                  input_sha256={name: entry['sha256'] for name, entry in manifest['inputs'].items()},
                  full_budget_iou75_tp=native_counts, final_input_pins_passed=True)
    with (output / 'report.json').open('x') as handle:
        json.dump(report, handle, indent=2)
    print('Completed cross-fitted calibration and all 2000 refit draws', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    pure_witness()
    if args.self_test:
        from mva_operating_points import self_test
        self_test()
        return
    if args.manifest is None or args.output is None:
        parser.error('--manifest and --output are required')
    args.output.mkdir()
    run(args.manifest, args.output)


if __name__ == '__main__':
    main()
