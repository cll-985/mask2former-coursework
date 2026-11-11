"""Signed single-region IoU changes at the frozen legacy endpoint."""

import numpy as np


def single_flip_utility(data, truth):
    """Return the anchored IoU change for flipping each v1 region in isolation.

    The result follows v1 component order. A missing anchor returns zeros.
    A supplied anchor must be nonempty and use the full original output grid.
    """
    base_bits = data['base_bits']
    if truth is None:
        return np.zeros(base_bits.shape, dtype=np.float64)

    truth = np.asarray(truth, dtype=bool)
    if truth.ndim != 2 or truth.shape != data['full_shape']:
        raise ValueError('Anchor truth must use the full original output grid')
    truth_area = np.count_nonzero(truth)
    if truth_area == 0:
        raise ValueError('An anchor truth must have positive full-image area')
    if not base_bits.size:
        return np.empty(0, dtype=np.float64)

    y0, y1, x0, x1 = data['crop_bbox']
    truth_crop = truth[y0:y1, x0:x1]
    endpoint = data['whole_masks'][data['endpoint_index']]
    intersection = np.count_nonzero(endpoint & truth_crop)
    union = truth_area + np.count_nonzero(endpoint & ~truth_crop)

    true_areas = np.concatenate([
        np.bincount(labels[truth_crop], minlength=count + 1)[1:]
        for labels, count in zip(data['labels_by_side'], data['counts'])
    ]).astype(np.int64, copy=False)
    false_areas = data['areas'] - true_areas
    direction = np.where(base_bits, -1, 1)
    flipped_union = union + direction * false_areas
    # Difference of the two IoU fractions, with integer cancellation for ties.
    delta_numerator = direction * (union * true_areas - intersection * false_areas)
    return delta_numerator / (float(union) * flipped_union)


def self_test():
    """Compare against full-mask reconstruction and exact IoU on synthetic grids."""
    from fractions import Fraction

    from offline_tools.mva_region_binary_features import extract, reconstruct

    def full_iou(mask, truth):
        intersection = int(np.count_nonzero(mask & truth))
        union = int(np.count_nonzero(mask | truth))
        return Fraction(intersection, union)

    empty = np.zeros((9, 10), dtype=bool)
    a, b, truth = empty.copy(), empty.copy(), empty.copy()
    a[4, 4] = b[4, 4] = True
    a[3, 2] = a[3, 6] = b[6, 2] = b[6, 6] = True
    truth[4, 4] = truth[3, 2] = truth[6, 2] = truth[8, 9] = True
    tie_a, tie_b, tie_truth = empty.copy(), empty.copy(), empty.copy()
    tie_a[2, 2] = tie_b[2, 2] = True
    tie_b[3, 2:6] = True
    tie_truth[2, 2] = tie_truth[3, 2] = tie_truth[0, 0] = True
    outside = empty.copy()
    outside[0, 0] = True
    toy_cases = [(a, b, truth), (tie_a, tie_b, tie_truth), (a, a, truth),
                 (empty, empty, truth), (a, empty, truth), (empty, b, truth),
                 (a, b, outside)]

    rng = np.random.default_rng(20261104)
    random_cases = []
    for shape in ((1, 1), (1, 7), (8, 1), (9, 11)):
        for _ in range(16):
            ra, rb, rt = rng.random((3,) + shape) < .35
            rt[-1, -1] = True
            random_cases.append((ra, rb, rt))

    compared_cases = compared_flips = 0
    seen_direction_signs = set()
    for case_index, (a, b, truth) in enumerate(toy_cases + random_cases):
        for score_a, score_b in ((.8, .4), (.4, .8), (.6, .6)):
            data = extract(a, b, score_a, score_b, case_index % 6 + 1)
            base_bits = data['base_bits']
            endpoint = a if score_a > score_b else (b if score_a < score_b else a | b)
            np.testing.assert_array_equal(reconstruct(data, base_bits), endpoint)
            assert len(base_bits) == sum(data['counts'])
            baseline_iou = full_iou(endpoint, truth)
            expected = []
            for index, kept in enumerate(base_bits):
                flipped = base_bits.copy()
                flipped[index] = not kept
                flipped_mask = reconstruct(data, flipped)
                assert np.count_nonzero(endpoint ^ flipped_mask) == data['areas'][index]
                delta = float(full_iou(flipped_mask, truth) - baseline_iou)
                expected.append(delta)
                seen_direction_signs.add((bool(kept), np.sign(delta)))
                if case_index == 1:
                    assert delta == 0 and baseline_iou == Fraction(1, 3)
            actual = single_flip_utility(data, truth)
            assert actual.dtype == np.float64 and actual.shape == base_bits.shape
            np.testing.assert_array_equal(actual, expected)
            no_anchor = single_flip_utility(data, None)
            assert no_anchor.dtype == np.float64
            np.testing.assert_array_equal(no_anchor, np.zeros_like(actual))
            compared_cases += 1
            compared_flips += len(base_bits)
    assert len(random_cases) == 64 and compared_cases == 3 * (len(toy_cases) + 64)
    assert seen_direction_signs == {(kept, sign) for kept in (False, True) for sign in (-1, 0, 1)}
    return dict(status='passed', random_masks=len(random_cases), toy_cases=len(toy_cases),
                compared_cases=compared_cases, compared_flips=compared_flips)


if __name__ == '__main__':
    import json

    print(json.dumps(self_test(), sort_keys=True))
