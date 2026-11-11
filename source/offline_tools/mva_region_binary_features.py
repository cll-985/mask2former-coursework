"""Frozen binary geometry and teacher labels; no fitting or real-data access.

``extract`` is GT independent. Spatial arrays in its result are cropped to the
inclusive union bounding box, represented as half-open (y0, y1, x0, x1).
``reconstruct`` and ``whole_reconstruct`` restore the original output grid.
The six category columns use the *predicted* category_id, always 1 through 6.
"""
import math

import numpy as np
from scipy import ndimage


PAIR_FEATURE_NAMES = (
    'score_a', 'score_b', 'score_a_minus_b', 'mask_iou',
    'log1p_area_a', 'log1p_area_b', 'log_area_ratio_a_b',
    'log1p_components_a', 'log1p_components_b', 'endpoint_side')
REGION_EXTRA_NAMES = (
    'source_side', 'area_over_union', 'area_over_xor', 'bbox_occupancy',
    'perimeter4_over_sqrt_area', 'touch4_core_over_perimeter4',
    'union_bbox_centroid_x', 'union_bbox_centroid_y')
SIDE_SUMMARY_NAMES = (
    'max_component_area_over_union', 'area_weighted_bbox_occupancy',
    'area_weighted_perimeter4_over_sqrt_area',
    'area_weighted_touch4_core_over_perimeter4',
    'area_weighted_union_bbox_centroid_x',
    'area_weighted_union_bbox_centroid_y')
CLASS_FEATURE_NAMES = tuple('predicted_category_{}'.format(i) for i in range(1, 7))
REGION_FEATURE_NAMES = PAIR_FEATURE_NAMES + REGION_EXTRA_NAMES + CLASS_FEATURE_NAMES
WHOLE_FEATURE_NAMES = (PAIR_FEATURE_NAMES +
                       tuple(side + '_' + name for side in ('a', 'b')
                             for name in SIDE_SUMMARY_NAMES) + CLASS_FEATURE_NAMES)
FEATURE_SCHEMA = dict(
    region=list(REGION_FEATURE_NAMES), whole=list(WHOLE_FEATURE_NAMES),
    region_numeric_columns=18, whole_numeric_columns=22,
    whole_menu=['A', 'B', 'union', 'intersection'],
    component_order='A-only then B-only; each side first raster occurrence',
    connectivity=8, perimeter_connectivity=4,
    crop_bbox_order=['y0', 'y1_exclusive', 'x0', 'x1_exclusive'],
    centroid='(mean_coordinate - union_bbox_min + .5) / union_bbox_size',
    empty_side_summary='six zeros', feature_gt_access=False)


def _label_in_raster_order(mask):
    labels, count = ndimage.label(
        np.ascontiguousarray(mask), structure=np.ones((3, 3), dtype=np.uint8))
    count = int(count)
    if count:
        # Canonicalize explicitly rather than depend on a CCL library's IDs.
        flat = np.flatnonzero(labels.ravel())
        first = np.full(count + 1, labels.size, dtype=np.int64)
        np.minimum.at(first, labels.ravel()[flat], flat)
        order = np.argsort(first[1:], kind='mergesort') + 1
        if not np.array_equal(order, np.arange(1, count + 1)):
            mapping = np.zeros(count + 1, dtype=labels.dtype)
            mapping[order] = np.arange(1, count + 1)
            labels = mapping[labels]
    return labels, count


def _side_geometry(labels, count, core, union_area, xor_area, side):
    """Vectorized per-label geometry, without constructing K full masks."""
    if count == 0:
        return np.empty(0, dtype=np.int64), np.empty((0, 8), dtype=np.float64)
    areas = np.bincount(labels.ravel(), minlength=count + 1)[1:].astype(np.int64)
    boxes = ndimage.find_objects(labels, max_label=count)
    bbox_areas = np.asarray([(box[0].stop - box[0].start) *
                             (box[1].stop - box[1].start) for box in boxes], dtype=np.float64)
    padded_labels = np.pad(labels, 1, mode='constant')
    padded_core = np.pad(core, 1, mode='constant')
    perimeter = np.zeros(count + 1, dtype=np.int64)
    touch = np.zeros(count + 1, dtype=np.int64)
    for neighbor, neighbor_core in (
            (padded_labels[:-2, 1:-1], padded_core[:-2, 1:-1]),
            (padded_labels[2:, 1:-1], padded_core[2:, 1:-1]),
            (padded_labels[1:-1, :-2], padded_core[1:-1, :-2]),
            (padded_labels[1:-1, 2:], padded_core[1:-1, 2:])):
        perimeter += np.bincount(labels[labels != neighbor], minlength=count + 1)
        touch += np.bincount(labels[neighbor_core], minlength=count + 1)
    perimeter, touch = perimeter[1:], touch[1:]
    assert np.all(perimeter > 0) and np.all(touch <= perimeter)
    yy, xx = np.nonzero(labels)
    ids = labels[yy, xx]
    centroid_x = np.bincount(ids, weights=xx, minlength=count + 1)[1:] / areas
    centroid_y = np.bincount(ids, weights=yy, minlength=count + 1)[1:] / areas
    features = np.column_stack((
        np.full(count, side, dtype=np.float64), areas / float(union_area),
        areas / float(xor_area), areas / bbox_areas,
        perimeter / np.sqrt(areas), touch / perimeter,
        (centroid_x + .5) / labels.shape[1], (centroid_y + .5) / labels.shape[0]))
    return areas, features


def extract(a, b, score_a, score_b, category_id):
    """Return 24D region and 28D whole features on the original binary grid.

    All spatial result arrays are union crops. Empty union uses (0, 0, 0, 0)
    and empty (0, 0) arrays; K=0 is a mandatory no-op for either learner.
    A and B retain their original seed identity regardless of score or GT.
    """
    a, b = np.asarray(a, dtype=bool), np.asarray(b, dtype=bool)
    if a.ndim != 2 or a.shape != b.shape or any(size <= 0 for size in a.shape):
        raise ValueError('A/B must have the same nonempty 2D output grid')
    score_a, score_b = float(score_a), float(score_b)
    if not math.isfinite(score_a) or not math.isfinite(score_b):
        raise ValueError('Scores must be finite original COCO JSON scores')
    if category_id not in range(1, 7):
        raise ValueError('Predicted category_id must be one of 1..6')
    category_id = int(category_id)
    full_shape = a.shape
    full_union = a | b
    yy, xx = np.nonzero(full_union)
    if yy.size:
        crop_bbox = int(yy.min()), int(yy.max()) + 1, int(xx.min()), int(xx.max()) + 1
    else:
        crop_bbox = 0, 0, 0, 0
    y0, y1, x0, x1 = crop_bbox
    aa = np.array(a[y0:y1, x0:x1], dtype=bool, order='C', copy=True)
    bb = np.array(b[y0:y1, x0:x1], dtype=bool, order='C', copy=True)
    union, core = aa | bb, aa & bb
    area_a, area_b, area_u, area_c = [int(mask.sum()) for mask in (aa, bb, union, core)]
    area_x = area_u - area_c
    if area_u:
        labeled = [_label_in_raster_order(mask) for mask in (aa & ~bb, bb & ~aa)]
    else:
        labeled = [(np.zeros((0, 0), dtype=np.int32), 0) for _ in range(2)]
    labels_by_side = tuple(value[0] for value in labeled)
    counts = tuple(value[1] for value in labeled)
    endpoint_side = 1 if score_a > score_b else (-1 if score_a < score_b else 0)
    endpoint_index = 0 if endpoint_side == 1 else (1 if endpoint_side == -1 else 2)
    pair = np.asarray((
        score_a, score_b, score_a - score_b, area_c / float(area_u) if area_u else 0.,
        math.log1p(area_a), math.log1p(area_b), math.log((area_a + 1.) / (area_b + 1.)),
        math.log1p(counts[0]), math.log1p(counts[1]), endpoint_side), dtype=np.float64)
    side_data = [_side_geometry(labels, count, core, area_u, area_x, side)
                 for labels, count, side in zip(labels_by_side, counts, (1, -1))]
    areas = np.concatenate([value[0] for value in side_data])
    region_geometry = np.concatenate([value[1] for value in side_data], axis=0)
    one_hot = np.zeros(6, dtype=np.float64)
    one_hot[category_id - 1] = 1.
    k = len(areas)
    region_features = np.concatenate((
        np.tile(pair, (k, 1)), region_geometry, np.tile(one_hot, (k, 1))), axis=1)
    summaries = []
    for side_areas, geometry in side_data:
        if side_areas.size:
            weights = side_areas / float(side_areas.sum())
            summaries.extend([float(side_areas.max()) / area_u] +
                             list(np.sum(geometry[:, 3:] * weights[:, None], axis=0)))
        else:
            summaries.extend([0.] * 6)
    whole_features = np.concatenate((pair, np.asarray(summaries), one_hot))
    base_bits = np.concatenate((
        np.full(counts[0], endpoint_side != -1, dtype=bool),
        np.full(counts[1], endpoint_side != 1, dtype=bool)))
    assert region_features.shape == (k, 24) and whole_features.shape == (28,)
    assert int(areas.sum()) == area_x
    assert np.all(np.isfinite(region_features)) and np.all(np.isfinite(whole_features))
    return dict(region_features=region_features, whole_features=whole_features,
                areas=areas, labels_by_side=labels_by_side, base_bits=base_bits,
                core=core, whole_masks=(aa, bb, union, core), counts=counts,
                crop_bbox=crop_bbox, offset=(y0, x0), full_shape=full_shape,
                endpoint_side=endpoint_side, endpoint_index=endpoint_index,
                union_area=area_u, xor_area=area_x)


def _checked_bits(data, bits):
    bits = np.asarray(bits)
    if bits.shape != data['base_bits'].shape or not np.all((bits == 0) | (bits == 1)):
        raise ValueError('One whole-component binary decision is required per region')
    return bits.astype(bool, copy=False)


def _assemble_crop(data, bits):
    bits = _checked_bits(data, bits)
    crop = data['core'].copy()
    offset = 0
    for labels, count in zip(data['labels_by_side'], data['counts']):
        selected = np.zeros(count + 1, dtype=bool)
        selected[1:] = bits[offset:offset + count]
        crop |= selected[labels]
        offset += count
    return crop


def _restore(data, crop):
    result = np.zeros(data['full_shape'], dtype=bool)
    y0, y1, x0, x1 = data['crop_bbox']
    result[y0:y1, x0:x1] = crop
    return result


def reconstruct(data, bits):
    """Retain core and selected components, restoring the full output grid."""
    return _restore(data, _assemble_crop(data, bits))


def whole_reconstruct(data, index):
    """Restore a fixed-menu choice (0=A, 1=B, 2=union, 3=intersection)."""
    if index not in range(4):
        raise ValueError('Whole menu index must be 0, 1, 2, or 3')
    return _restore(data, data['whole_masks'][int(index)])


def _bits_from_crop(data, crop):
    """Assert that a teacher mask takes each entire component or none of it."""
    assert np.array_equal(crop & data['core'], data['core'])
    assert not np.any(crop & ~data['whole_masks'][2])
    bits = []
    offset = 0
    for labels, count in zip(data['labels_by_side'], data['counts']):
        kept = np.bincount(labels[crop], minlength=count + 1)[1:]
        areas = data['areas'][offset:offset + count]
        assert np.all((kept == 0) | (kept == areas)), 'Partial-component teacher label'
        bits.extend(kept == areas)
        offset += count
    return np.asarray(bits, dtype=bool)


def target(data, truth_or_none):
    """Return (region_bits, whole_index), reusing the already extracted labels.

    None means no positive anchor and preserves the exact-score-tie union
    endpoint. If several whole menu masks equal that endpoint, label the first.
    A supplied truth is the full-image chosen GT; its total area is never
    cropped when computing IoU. This function cannot alter feature columns.
    """
    if truth_or_none is None:
        endpoint = data['whole_masks'][data['endpoint_index']]
        whole_index = next(i for i, mask in enumerate(data['whole_masks'])
                           if np.array_equal(mask, endpoint))
        bits = data['base_bits'].copy()
        assert np.array_equal(_bits_from_crop(data, endpoint), bits)
        return bits, whole_index
    truth = np.asarray(truth_or_none, dtype=bool)
    if truth.shape != data['full_shape'] or truth.ndim != 2:
        raise ValueError('Teacher truth must use the full original output grid')
    g = int(truth.sum())
    if g <= 0:
        raise ValueError('An anchor truth must have positive full-image area')
    y0, y1, x0, x1 = data['crop_bbox']
    gg = truth[y0:y1, x0:x1]
    menu = [(int((mask & gg).sum()), g + int((mask & ~gg).sum()))
            for mask in data['whole_masks']]
    whole_index = 0
    for index in range(1, 4):
        if menu[index][0] * menu[whole_index][1] > menu[whole_index][0] * menu[index][1]:
            whole_index = index
    stats = []
    offset = 0
    for labels, count in zip(data['labels_by_side'], data['counts']):
        true_areas = np.bincount(labels[gg], minlength=count + 1)[1:]
        stats.extend((int(t), int(area - t))
                     for area, t in zip(data['areas'][offset:offset + count], true_areas))
        offset += count
    # Reuse the frozen exact integer optimum, including its stable prefix ties.
    from offline_tools.mva_region_capacity_probe import best_component_prefix
    t0 = int((data['core'] & gg).sum())
    d0 = g + int((data['core'] & ~gg).sum())
    cn, cd, selected = best_component_prefix(t0, d0, stats)
    bits = np.zeros(len(stats), dtype=bool)
    bits[selected] = True
    crop = _assemble_crop(data, bits)
    assert np.array_equal(_bits_from_crop(data, crop), bits)
    n, d = int((crop & gg).sum()), g + int((crop & ~gg).sum())
    assert n * cd == cn * d, 'Reused labels disagree with exact teacher optimum'
    wn, wd = menu[whole_index]
    assert wn * cd <= cn * wd
    return bits, whole_index


def _brute_features(a, b, score_a, score_b, category_id):
    """Independent tiny-grid reference: Python raster flood and edge counting."""
    h, w = a.shape
    union, core = a | b, a & b
    points_u = [(y, x) for y in range(h) for x in range(w) if union[y, x]]
    if points_u:
        y0, y1 = min(y for y, x in points_u), max(y for y, x in points_u) + 1
        x0, x1 = min(x for y, x in points_u), max(x for y, x in points_u) + 1
    else:
        y0 = y1 = x0 = x1 = 0
    labels_by_side, regions_by_side = [], []
    for membership in (a & ~b, b & ~a):
        labels = np.zeros((h, w), dtype=np.int32)
        regions = []
        for y in range(h):
            for x in range(w):
                if not membership[y, x] or labels[y, x]:
                    continue
                value = len(regions) + 1
                labels[y, x] = value
                todo, points = [(y, x)], []
                while todo:
                    yy, xx = todo.pop()
                    points.append((yy, xx))
                    for dy in (-1, 0, 1):
                        for dx in (-1, 0, 1):
                            ny, nx = yy + dy, xx + dx
                            if (0 <= ny < h and 0 <= nx < w and membership[ny, nx]
                                    and not labels[ny, nx]):
                                labels[ny, nx] = value
                                todo.append((ny, nx))
                regions.append(points)
        labels_by_side.append(labels)
        regions_by_side.append(regions)
    counts = tuple(len(regions) for regions in regions_by_side)
    area_a, area_b, area_u, area_c = [sum(bool(z) for z in mask.ravel())
                                    for mask in (a, b, union, core)]
    area_x = area_u - area_c
    endpoint_side = 1 if score_a > score_b else (-1 if score_a < score_b else 0)
    p = [score_a, score_b, score_a - score_b, area_c / area_u if area_u else 0.,
         math.log(1 + area_a), math.log(1 + area_b),
         math.log((1 + area_a) / (1 + area_b)),
         math.log(1 + counts[0]), math.log(1 + counts[1]), endpoint_side]
    one_hot = [float(category_id == i) for i in range(1, 7)]
    rows, areas, summaries = [], [], []
    for side, regions in zip((1, -1), regions_by_side):
        side_r, side_areas = [], []
        for points in regions:
            points_set = set(points)
            area, perimeter, touch = len(points), 0, 0
            for y, x in points:
                for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    ny, nx = y + dy, x + dx
                    if (ny, nx) not in points_set:
                        perimeter += 1
                        if 0 <= ny < h and 0 <= nx < w and core[ny, nx]:
                            touch += 1
            ys, xs = [y for y, x in points], [x for y, x in points]
            bbox = (max(ys) - min(ys) + 1) * (max(xs) - min(xs) + 1)
            r = [side, area / area_u, area / area_x, area / bbox,
                 perimeter / math.sqrt(area), touch / perimeter,
                 (sum(xs) / area - x0 + .5) / (x1 - x0),
                 (sum(ys) / area - y0 + .5) / (y1 - y0)]
            rows.append(p + r + one_hot)
            areas.append(area)
            side_r.append(r)
            side_areas.append(area)
        if side_areas:
            summaries.extend([max(side_areas) / area_u] +
                             [sum(area * r[j] for area, r in zip(side_areas, side_r)) /
                              sum(side_areas) for j in range(3, 8)])
        else:
            summaries.extend([0.] * 6)
    return (np.asarray(rows, dtype=np.float64).reshape((-1, 24)),
            np.asarray(p + summaries + one_hot), np.asarray(areas), labels_by_side)


def self_test():
    """Synthetic only: independent geometry and frozen legacy reconstruction."""
    from offline_tools.mva_region_ap_probe import assembled_masks
    from offline_tools.mva_region_capacity_probe import membership_components
    rng = np.random.RandomState(20261103)
    fixtures = []
    for shape in ((1, 1), (1, 7), (8, 1), (9, 11)):
        for _ in range(16):
            a, b, truth = [rng.rand(*shape) < .35 for _ in range(3)]
            truth[-1, -1] = True
            fixtures.append((a, b, truth))
    z = np.zeros((10, 12), dtype=bool)
    a, b, truth = z.copy(), z.copy(), z.copy()
    a[3, 4] = a[4, 5] = a[6, 7] = True  # diagonal 8-connectivity and bbox offset
    b[3, 4] = b[3, 5] = b[7, 8] = True
    truth[3, 4] = truth[4, 5] = truth[9, 11] = True  # full GT extends outside crop
    fixtures.extend([(a, b, truth), (a, a, truth), (z, z, truth),
                     (a, z, truth), (z, b, truth), (a | b, a & b, truth)])
    # A region wrapping a core has multiple touching edges; image outer edges
    # and a hole both contribute to perimeter. The brute loop checks each edge.
    ring, center = np.ones((5, 5), dtype=bool), np.zeros((5, 5), dtype=bool)
    center[1:4, 1:4] = True
    fixtures.append((ring, center, ring.copy()))
    compared = 0
    for fixture_index, (a, b, truth) in enumerate(fixtures):
        for score_a, score_b in ((.8, .4), (.4, .8), (.6, .6)):
            category_id = fixture_index % 6 + 1
            saved_inputs = tuple(x.copy() for x in (a, b, truth))
            data = extract(a, b, score_a, score_b, category_id)
            expected_r, expected_w, expected_areas, expected_labels = _brute_features(
                a, b, score_a, score_b, category_id)
            np.testing.assert_allclose(data['region_features'], expected_r, rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(data['whole_features'], expected_w, rtol=1e-12, atol=1e-12)
            assert np.array_equal(data['areas'], expected_areas)
            for labels, expected in zip(data['labels_by_side'], expected_labels):
                assert np.array_equal(_restore(data, labels), expected > 0)
                restored = np.zeros(a.shape, dtype=labels.dtype)
                y0, y1, x0, x1 = data['crop_bbox']
                restored[y0:y1, x0:x1] = labels
                assert np.array_equal(restored, expected)
            assert all(np.array_equal(whole_reconstruct(data, i), mask)
                       for i, mask in enumerate((a, b, a | b, a & b)))
            endpoint = a if score_a > score_b else (b if score_a < score_b else a | b)
            assert np.array_equal(reconstruct(data, data['base_bits']), endpoint)
            no_anchor_bits, no_anchor_whole = target(data, None)
            assert np.array_equal(no_anchor_bits, data['base_bits'])
            assert np.array_equal(whole_reconstruct(data, no_anchor_whole), endpoint)
            assert no_anchor_whole == next(i for i, m in enumerate((a, b, a | b, a & b))
                                           if np.array_equal(m, endpoint))
            feature_snapshot = data['region_features'].copy(), data['whole_features'].copy()
            bits, whole_index = target(data, truth)
            legacy = assembled_masks(a, b, truth)
            assert np.array_equal(reconstruct(data, bits), legacy['component'])
            assert np.array_equal(whole_reconstruct(data, whole_index), legacy['whole'])
            old_stats, old_labels = membership_components(a, b, truth)
            assert len(old_stats) == len(bits)
            for crop_labels, old in zip(data['labels_by_side'], old_labels):
                y0, y1, x0, x1 = data['crop_bbox']
                assert np.array_equal(crop_labels, old[y0:y1, x0:x1])
            alternative_truth = ~truth
            if not alternative_truth.any():
                alternative_truth[0, 0] = True
            target(data, alternative_truth)
            assert all(np.array_equal(before, after) for before, after in
                       zip(feature_snapshot, (data['region_features'], data['whole_features'])))
            assert all(np.array_equal(before, after) for before, after in
                       zip(saved_inputs, (a, b, truth)))
            if not data['xor_area']:
                assert data['region_features'].shape == (0, 24) and not bits.size
                assert np.array_equal(reconstruct(data, bits), a)
                assert all(np.array_equal(whole_reconstruct(data, i), a) for i in range(4))
            compared += 1
    # Identical menu masks use the first index even for union endpoint ties.
    equal = extract(a, a, .5, .5, 1)
    assert target(equal, None)[1] == 0 and target(equal, truth)[1] == 0
    # With a disjoint chosen GT, every IoU ties at zero: first whole option A.
    aa = z.copy(); bb = z.copy(); outside = z.copy()
    aa[2, 2] = True; bb[3, 3] = True; outside[9, 11] = True
    disjoint = extract(aa, bb, .5, .5, 6)
    bits, index = target(disjoint, outside)
    assert index == 0 and not bits.any()
    assert target(disjoint, None)[0].all() and target(disjoint, None)[1] == 2
    # Deliberately cut one connected component: the teacher validator must fail.
    cut_a = z.copy(); cut_a[2, 2:4] = True
    cut_data = extract(cut_a, z, .8, .4, 1)
    cut_crop = cut_data['core'].copy(); cut_crop[0, 0] = True
    try:
        _bits_from_crop(cut_data, cut_crop)
    except AssertionError:
        pass
    else:
        raise AssertionError('Partial region label was accepted')
    return dict(status='passed', synthetic_feature_and_teacher_cases=compared,
                independent_raster_flood_and_pixel_edges=True,
                legacy_assembled_masks_exact=True, gt_independent_features=True,
                zero_xor_and_empty_union=True, exact_score_union_and_stable_ties=True,
                no_anchor_endpoint_fallback=True, no_real_inputs=True)


if __name__ == '__main__':
    import json
    print(json.dumps(self_test(), sort_keys=True))
