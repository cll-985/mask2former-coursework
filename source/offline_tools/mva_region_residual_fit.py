"""Crossfit component flips relative to the frozen higher-score endpoint."""

import json
from pathlib import Path

import numpy as np

from offline_tools.mva_region_binary_features import FEATURE_SCHEMA
from offline_tools.mva_region_binary_fit import N_FOLDS, fit, predict, weights


REGION_FEATURES = FEATURE_SCHEMA['region']
NUMERIC_DIMS = FEATURE_SCHEMA['region_numeric_columns']
ENDPOINT_SIDE = REGION_FEATURES.index('endpoint_side')
SOURCE_SIDE = REGION_FEATURES.index('source_side')


def default_bits(x):
    """Keep the higher-score side's components, or both sides on a score tie."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != len(REGION_FEATURES):
        raise ValueError('Expected the frozen region feature layout')
    endpoint = x[:, ENDPOINT_SIDE]
    source = x[:, SOURCE_SIDE]
    if not np.isin(endpoint, (-1, 0, 1)).all() or not np.isin(source, (-1, 1)).all():
        raise ValueError('Expected finite endpoint and component side signs')
    return (endpoint == 0) | (endpoint == source)


def _checked_bits(bits, shape, name):
    bits = np.asarray(bits)
    if bits.shape != shape or not ((bits == 0) | (bits == 1)).all():
        raise ValueError('{} must contain one binary value per region'.format(name))
    return bits.astype(bool, copy=False)


def to_keep(flip_prob, base):
    """Apply flips strictly above one half to the endpoint's keep bits."""
    flip_prob = np.asarray(flip_prob, dtype=np.float64)
    if (flip_prob.ndim != 1 or not np.isfinite(flip_prob).all()
            or not ((flip_prob >= 0) & (flip_prob <= 1)).all()):
        raise ValueError('Expected a vector of finite flip probabilities in [0, 1]')
    base = _checked_bits(base, flip_prob.shape, 'Endpoint keep bits')
    return np.logical_xor(base, flip_prob > .5)


def fit_oof(data, folds, folder, label):
    """Fit the frozen binary estimator to teacher XOR endpoint on training rows."""
    folder = Path(folder)
    x, y, images, candidates, areas = (np.asarray(data[key]) for key in
                                       ('rx', 'ry', 'ri', 'rc', 'ra'))
    base = default_bits(x)
    teacher = _checked_bits(y, base.shape, 'Teacher keep bits')
    if any(values.shape != base.shape for values in (images, candidates, areas)):
        raise ValueError('Expected an image, candidate and area per region')
    targets = np.logical_xor(teacher, base)
    row_folds = np.asarray([folds[image] for image in images])
    if set(row_folds) != set(range(N_FOLDS)):
        raise ValueError('Each frozen fold must have held-out region rows')

    probabilities = np.full(len(x), np.nan, dtype=np.float64)
    logs = []
    for fold in range(N_FOLDS):
        test = row_folds == fold
        train = ~test
        train_ids = set(map(int, images[train]))
        test_ids = set(map(int, images[test]))
        if train_ids & test_ids:
            raise ValueError('An image appears in both train and held-out rows')
        train_weights = weights(images[train], candidates[train], areas[train])
        model = fit(x[train], targets[train], train_weights, NUMERIC_DIMS, 2)
        probabilities[test] = predict(model, x[test])
        record = dict(fold=fold, model='residual', target='teacher_keep XOR endpoint_keep',
                      train_images=sorted(train_ids), test_images=sorted(test_ids),
                      train_rows=int(train.sum()), test_rows=int(test.sum()))
        path = folder / '{}_residual_fold{}.json'.format(label, fold)
        with path.open('x', encoding='utf-8') as handle:
            json.dump(dict(record, parameters=model), handle, indent=2, sort_keys=True,
                      allow_nan=False)
            handle.write('\n')
        logs.append(record)
        print('FIT_COMPLETE {} residual fold={} train={} test={}'.format(
            label, fold, record['train_rows'], record['test_rows']), flush=True)

    keep = to_keep(probabilities, base)
    with (folder / '{}_residual_oof.npz'.format(label)).open('xb') as handle:
        np.savez_compressed(handle, flip_probability=probabilities, base_bits=base,
                            keep_bits=keep)
    return probabilities, base, keep, logs


def self_test():
    """Exercise the residual logic with synthetic masks and five image folds."""
    from tempfile import TemporaryDirectory

    from offline_tools.mva_region_binary_features import extract, reconstruct, target
    from offline_tools.mva_region_binary_fit import make_folds

    core = np.zeros((12, 12), dtype=bool)
    core[4:7, 4:7] = True
    a, b = core.copy(), core.copy()
    a[3, 4] = True
    b[7, 6] = True
    for score_a, score_b, expected in ((.8, .7, a), (.7, .8, b), (.8, .8, a | b)):
        features = extract(a, b, score_a, score_b, 1)
        base = default_bits(features['region_features'])
        np.testing.assert_array_equal(base, features['base_bits'])
        np.testing.assert_array_equal(reconstruct(features, to_keep([.5, .5], base)),
                                      expected)

    base = np.array([0, 0, 1, 1], dtype=bool)
    teacher = np.array([0, 1, 0, 1], dtype=bool)
    flips = np.logical_xor(teacher, base)
    np.testing.assert_array_equal(flips, [False, True, True, False])
    np.testing.assert_array_equal(to_keep(flips, base), teacher)
    below, above = np.nextafter(.5, 0.), np.nextafter(.5, 1.)
    np.testing.assert_array_equal(to_keep([below, .5, above, .5], base), [0, 0, 0, 1])

    edit_b = b.copy()
    edit_b[7:10, 4:7] = True
    examples = (extract(a, b, .8, .7, 1), extract(a, edit_b, .8, .7, 1))
    x = np.concatenate([features['region_features'] for features in examples])
    areas = np.concatenate([features['areas'] for features in examples])
    edit_teacher = np.array([False, True])
    np.testing.assert_array_equal(reconstruct(examples[1], edit_teacher), edit_b)
    y = np.concatenate((target(examples[0], None)[0], edit_teacher))
    images = np.arange(1, 11, dtype=np.int64)
    data = dict(rx=np.tile(x, (len(images), 1)), ry=np.tile(y, len(images)),
                ri=np.repeat(images, len(x)), rc=np.tile([0, 0, 1, 1], len(images)),
                ra=np.tile(areas, len(images)))
    with TemporaryDirectory(dir=str(Path.cwd())) as folder:
        probability, base, keep, logs = fit_oof(data, make_folds(images), folder, 'synthetic')
        with np.load(str(Path(folder) / 'synthetic_residual_oof.npz')) as saved:
            for key, values in (('flip_probability', probability), ('base_bits', base),
                                ('keep_bits', keep)):
                np.testing.assert_array_equal(saved[key], values)
        for fold in range(N_FOLDS):
            path = Path(folder) / 'synthetic_residual_fold{}.json'.format(fold)
            record = json.loads(path.read_text(encoding='utf-8'))
            assert record['parameters']['optimizer']['success']
    no_edit = data['rc'] == 0
    assert (probability[no_edit] < .5).all()
    assert (probability[~no_edit] > .5).all()
    np.testing.assert_array_equal(keep[no_edit], base[no_edit])
    np.testing.assert_array_equal(keep, data['ry'])
    return dict(status='passed', synthetic_only=True, score_swap_no_op=True,
                xor_targets=True, exact_tie_no_op=True, learned_no_edit=True,
                learned_positive_edit=True, converged_folds=len(logs))
