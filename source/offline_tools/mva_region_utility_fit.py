"""Crossfit signed single-component utility and match residual flip counts."""

import json
from pathlib import Path

import numpy as np

from offline_tools.mva_region_binary_features import FEATURE_SCHEMA
from offline_tools.mva_region_binary_fit import (
    BIAS_L2,
    NONBIAS_L2,
    N_FOLDS,
    _matrix,
    _standardize,
    weights,
)


N_FEATURES = len(FEATURE_SCHEMA['region'])
NUMERIC_DIMS = FEATURE_SCHEMA['region_numeric_columns']


def fit(x, utility, row_weights):
    """Fit weighted ridge on the supplied training rows in float64."""
    x = _matrix(x)
    if not len(x) or x.shape[1] != N_FEATURES:
        raise ValueError('Expected nonempty training rows in the frozen region layout')
    utility = np.asarray(utility, dtype=np.float64)
    if utility.shape != (len(x),) or not np.isfinite(utility).all():
        raise ValueError('Expected one finite signed utility per training row')
    row_weights = np.asarray(row_weights, dtype=np.float64)
    if (row_weights.shape != utility.shape or not np.isfinite(row_weights).all()
            or not (row_weights > 0).all()
            or not np.isclose(row_weights.sum(), 1., rtol=1e-12, atol=1e-12)):
        raise ValueError('Training weights must be positive and sum to one')
    row_weights = row_weights / row_weights.sum()

    numeric = x[:, :NUMERIC_DIMS]
    mean = np.sum(row_weights[:, None] * numeric, axis=0)
    constant = np.all(numeric == numeric[0], axis=0)
    mean[constant] = numeric[0, constant]
    std = np.sqrt(np.sum(row_weights[:, None] * np.square(numeric - mean), axis=0))
    std[std == 0.] = 1.
    if not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise ValueError('Nonfinite training mean or standard deviation')

    design = np.empty((len(x), N_FEATURES + 1), dtype=np.float64)
    design[:, :-1] = _standardize(x, mean, std, NUMERIC_DIMS)
    design[:, -1] = 1.
    penalties = np.full(N_FEATURES + 1, NONBIAS_L2, dtype=np.float64)
    penalties[-1] = BIAS_L2
    weighted_transpose = design.T * row_weights
    normal = weighted_transpose @ design + np.diag(penalties)
    rhs = weighted_transpose @ utility
    parameters = np.linalg.solve(normal, rhs)
    if not np.isfinite(parameters).all():
        raise FloatingPointError('Ridge solve returned nonfinite parameters')
    residual = design @ parameters - utility
    objective = .5 * (row_weights @ np.square(residual)
                      + penalties @ np.square(parameters))
    return dict(
        schema_version=1, type='weighted_ridge', n_features=N_FEATURES,
        numeric_dims=NUMERIC_DIMS, mean=mean.tolist(), std=std.tolist(),
        coef=parameters[:-1].tolist(), intercept=float(parameters[-1]),
        regularization=dict(nonbias_l2=NONBIAS_L2, bias_l2=BIAS_L2),
        solver=dict(method='normal_equations', dtype='float64',
                    objective=float(objective),
                    gradient_inf_norm=float(np.abs(normal @ parameters - rhs).max())),
        training=dict(rows=len(x), weight_sum=float(row_weights.sum()),
                      zero_std_columns=np.flatnonzero(constant).tolist(),
                      weighted_utility_mean=float(row_weights @ utility),
                      weighted_utility_positive_rate=float(row_weights @ (utility > 0))),
    )


def predict(model, x):
    """Predict signed utility using the training means and standard deviations."""
    x = _matrix(x)
    if x.shape[1] != N_FEATURES:
        raise ValueError('Prediction layout differs from the frozen region layout')
    standardized = _standardize(x, model['mean'], model['std'], NUMERIC_DIMS)
    result = standardized @ np.asarray(model['coef'], dtype=np.float64) + model['intercept']
    if not np.isfinite(result).all():
        raise FloatingPointError('Nonfinite utility prediction')
    return result


def _top_k_bits(scores, row_indices, budget):
    """Return exactly K flips, breaking equal scores by frozen global row index."""
    if not 0 <= budget <= len(scores):
        raise ValueError('Flip budget must be between zero and the held-out row count')
    selected = np.zeros(len(scores), dtype=bool)
    if budget:
        order = np.lexsort((row_indices, -scores))
        selected[order[:budget]] = True
    return selected


def fit_oof(data, folds, utility, residual_p, folder, label):
    """Fit signed utility OOF and match each residual fold's count above .5."""
    folder = Path(folder)
    x, images, candidates, areas = (np.asarray(data[key]) for key in ('rx', 'ri', 'rc', 'ra'))
    x = _matrix(x)
    utility = np.asarray(utility, dtype=np.float64)
    residual_p = np.asarray(residual_p, dtype=np.float64)
    if x.shape[1] != N_FEATURES:
        raise ValueError('Expected the frozen region feature layout')
    if any(values.shape != (len(x),) for values in (images, candidates, areas, utility, residual_p)):
        raise ValueError('Expected image, candidate, area, utility and probability per region')
    if not np.isfinite(utility).all():
        raise ValueError('Expected finite signed utility targets')
    if not np.isfinite(residual_p).all() or not ((residual_p >= 0.) & (residual_p <= 1.)).all():
        raise ValueError('Expected finite residual probabilities in [0, 1]')
    row_folds = np.asarray([folds[image] for image in images])
    if set(row_folds) != set(range(N_FOLDS)):
        raise ValueError('Each frozen fold must have held-out region rows')

    utility_oof = np.full(len(x), np.nan, dtype=np.float64)
    flip_bits = np.zeros(len(x), dtype=bool)
    fold_records = []
    for fold in range(N_FOLDS):
        test = row_folds == fold
        train = ~test
        train_ids = set(map(int, images[train]))
        test_ids = set(map(int, images[test]))
        if train_ids & test_ids:
            raise ValueError('An image appears in both train and held-out rows')

        train_weights = weights(images[train], candidates[train], areas[train])
        model = fit(x[train], utility[train], train_weights)
        scores = predict(model, x[test])
        budget = int(np.count_nonzero(residual_p[test] > .5))
        selected = _top_k_bits(scores, np.flatnonzero(test), budget)
        utility_oof[test] = scores
        flip_bits[test] = selected

        # The held-out utilities and weights contribute only to the audit.
        test_utility = utility[test]
        test_weights = weights(images[test], candidates[test], areas[test])
        record = dict(
            fold=fold, model='utility', target='signed single-component utility',
            train_images=sorted(train_ids), test_images=sorted(test_ids),
            train_rows=int(train.sum()), test_rows=len(scores), budget=budget,
            selected_flips=int(selected.sum()),
            selected_negative_predictions=int(np.count_nonzero(selected & (scores < 0.))),
            selected_zero_predictions=int(np.count_nonzero(selected & (scores == 0.))),
            train_weighted_utility_mean=model['training']['weighted_utility_mean'],
            train_weighted_utility_positive_rate=model['training']['weighted_utility_positive_rate'],
            test_weighted_utility_mean=float(test_weights @ test_utility),
            test_weighted_utility_positive_rate=float(test_weights @ (test_utility > 0.)),
            test_weighted_prediction_mean=float(test_weights @ scores),
            test_weighted_prediction_positive_rate=float(test_weights @ (scores > 0.)),
        )
        path = folder / '{}_utility_fold{}.json'.format(label, fold)
        with path.open('x', encoding='utf-8') as handle:
            json.dump(dict(record, parameters=model), handle, indent=2, sort_keys=True,
                      allow_nan=False)
            handle.write('\n')
        fold_records.append(record)
        print('FIT_COMPLETE {} utility fold={} train={} test={} budget={} negative={}'.format(
            label, fold, record['train_rows'], record['test_rows'], budget,
            record['selected_negative_predictions']), flush=True)

    with (folder / '{}_utility_oof.npz'.format(label)).open('xb') as handle:
        np.savez_compressed(handle, utility_prediction=utility_oof, flip_bits=flip_bits)
    return utility_oof, flip_bits, fold_records


def self_test():
    """Check the numerical fit, deterministic budgets and image-disjoint OOF."""
    from tempfile import TemporaryDirectory

    from offline_tools.mva_region_binary_fit import make_folds

    rng = np.random.RandomState(20261103)
    images = np.repeat(np.arange(1, 11), 4)
    candidates = np.tile([0, 0, 1, 1], 10)
    areas = np.tile([1, 3, 6, 2], 10)
    x = rng.normal(size=(len(images), N_FEATURES))
    x[:, NUMERIC_DIMS - 1] = 7.
    x[:, NUMERIC_DIMS:] = np.eye(6)[np.arange(len(x)) % 6]
    original = x.copy()
    utility = .3 * x[:, 0] - .2 * x[:, 1] + .1 * x[:, -1] - .2
    row_weights = weights(images, candidates, areas)
    model = fit(x, utility, row_weights)
    mean = np.average(x[:, :NUMERIC_DIMS], weights=row_weights, axis=0)
    std = np.sqrt(np.average((x[:, :NUMERIC_DIMS] - mean) ** 2,
                             weights=row_weights, axis=0))
    std[-1] = 1.
    np.testing.assert_allclose(model['mean'], mean, atol=1e-14, rtol=1e-14)
    np.testing.assert_allclose(model['std'], std, atol=1e-14, rtol=1e-14)
    np.testing.assert_array_equal(x, original)
    design = np.column_stack(((x[:, :NUMERIC_DIMS] - mean) / std,
                              x[:, NUMERIC_DIMS:], np.ones(len(x))))
    penalty = np.sqrt([NONBIAS_L2] * N_FEATURES + [BIAS_L2])
    augmented = np.vstack((design * np.sqrt(row_weights)[:, None], np.diag(penalty)))
    response = np.concatenate((utility * np.sqrt(row_weights), np.zeros(len(penalty))))
    expected, _, _, _ = np.linalg.lstsq(augmented, response, rcond=None)
    np.testing.assert_allclose(model['coef'] + [model['intercept']], expected, atol=1e-12)
    np.testing.assert_allclose(predict(model, x), design @ expected, atol=1e-12)

    scores = np.array([-1., .2, .2, 0.])
    row_indices = np.array([8, 2, 0, 5])
    np.testing.assert_array_equal(_top_k_bits(scores, row_indices, 1), [0, 0, 1, 0])
    np.testing.assert_array_equal(_top_k_bits(scores, row_indices, 3), [0, 1, 1, 1])
    assert _top_k_bits(scores, row_indices, len(scores)).all()
    assert not _top_k_bits(scores, row_indices, 0).any()
    assert _top_k_bits(scores[:0], row_indices[:0], 0).size == 0

    data = dict(rx=x, ri=images, rc=candidates, ra=areas)
    folds = make_folds(images)
    row_folds = np.asarray([folds[image] for image in images])
    residual_p = np.full(len(x), .5)
    residual_p[row_folds == 1] = np.nextafter(.5, 1.)
    residual_p[(row_folds > 1) & (areas == 3)] = .9
    with TemporaryDirectory(dir=str(Path.cwd())) as folder:
        pred, selected, records = fit_oof(data, folds, utility, residual_p, folder, 'synthetic')
        for record in records:
            fold = record['fold']
            test = row_folds == fold
            train = ~test
            assert set(record['train_images']).isdisjoint(record['test_images'])
            assert record['budget'] == record['selected_flips'] == int(selected[test].sum())
            assert record['budget'] == int(np.count_nonzero(residual_p[test] > .5))
            assert record['selected_negative_predictions'] == int((selected[test] & (pred[test] < 0.)).sum())
            path = Path(folder) / 'synthetic_utility_fold{}.json'.format(fold)
            saved_model = json.loads(path.read_text(encoding='utf-8'))['parameters']
            train_weights = weights(images[train], candidates[train], areas[train])
            expected_mean = np.average(x[train, :NUMERIC_DIMS], weights=train_weights, axis=0)
            np.testing.assert_allclose(saved_model['mean'], expected_mean, atol=1e-14)
            np.testing.assert_array_equal(predict(saved_model, x[test]), pred[test])
        with np.load(str(Path(folder) / 'synthetic_utility_oof.npz')) as saved:
            np.testing.assert_array_equal(saved['utility_prediction'], pred)
            np.testing.assert_array_equal(saved['flip_bits'], selected)
    assert records[0]['budget'] == 0
    assert records[1]['selected_negative_predictions'] > 0
    return dict(status='passed', synthetic_only=True, weighted_ridge_reference=True,
                frozen_index_tie_break=True, exact_fold_budget=True,
                empty_budget=True, forced_negative=True,
                disjoint_image_folds=len(records))
