"""Frozen CPU estimators for the 20261103 binary-region crossfit diagnostic.

This module does not open prediction/GT files. Callers supply only the training
rows to ``weights`` and ``fit``; an empty-XOR candidate has no training row in
either task. Models are plain JSON-compatible dictionaries, never pickles.
"""

import hashlib
import json
import operator


FOLD_SEED = "20261103:"
N_FOLDS = 5
NONBIAS_L2 = 0.01
BIAS_L2 = 1e-6
OPTIMIZER_OPTIONS = {
    "maxiter": 500,
    "maxls": 20,
    "ftol": 1e-10,
    "gtol": 1e-6,
}
_LAYOUTS = {2: (24, 18), 4: (28, 22)}


def _image_id(value):
    """Accept integral IDs without silently truncating floats or strings."""
    if isinstance(value, bool):
        raise ValueError("An image ID must be an integer, not bool")
    try:
        return int(operator.index(value))
    except TypeError as exc:
        raise ValueError("An image ID must be an integer") from exc


def make_folds(image_ids):
    """Return the shared image-ID-to-fold mapping, independent of family/order.

    Duplicate IDs deliberately collapse to one image. Supply the complete frozen
    image manifest once and share the resulting mapping between both families.
    SHA256 is applied to UTF-8 ``'20261103:' + str(integer_image_id)``; digest
    bytes are sorted, then the sorted index modulo five is the fold.
    """
    ids = {_image_id(value) for value in image_ids}
    ordered = sorted(
        ids,
        key=lambda value: (
            hashlib.sha256((FOLD_SEED + str(value)).encode("utf-8")).digest(),
            value,
        ),
    )
    return {value: index % N_FOLDS for index, value in enumerate(ordered)}


def weights(image_ids, candidate_ids, areas=None):
    """Assign image/candidate equal weights to the supplied training rows.

    ``candidate_ids`` must be hashable and identify candidates within an image.
    A region candidate has one row per component and requires positive ``areas``;
    its row shares are area / sum(component areas). With ``areas=None`` the task
    is whole-menu classification and each candidate must have exactly one row.
    Callers must exclude held-out rows and all empty-XOR candidates beforehand.
    """
    import numpy as np

    images = [_image_id(value) for value in image_ids]
    candidates = list(candidate_ids)
    if not images or len(images) != len(candidates):
        raise ValueError("Weights require nonempty, equally sized row IDs")
    row_areas = None if areas is None else np.asarray(areas, dtype=np.float64)
    if row_areas is not None:
        if row_areas.shape != (len(images),):
            raise ValueError("Region areas must be one-dimensional, one per row")
        if not np.isfinite(row_areas).all() or not (row_areas > 0).all():
            raise ValueError("Region areas must be finite and strictly positive")

    groups = {}
    for row, (image, candidate) in enumerate(zip(images, candidates)):
        try:
            hash(candidate)
        except TypeError as exc:
            raise ValueError("Candidate IDs must be hashable") from exc
        groups.setdefault(image, {}).setdefault(candidate, []).append(row)
    result = np.zeros(len(images), dtype=np.float64)
    for image_candidates in groups.values():
        candidate_weight = 1.0 / (len(groups) * len(image_candidates))
        for rows in image_candidates.values():
            if row_areas is None:
                if len(rows) != 1:
                    raise ValueError("Whole-menu weights require one row per candidate")
                result[rows[0]] = candidate_weight
            else:
                local = row_areas[rows]
                # Scale first to avoid overflowing the sum of finite areas.
                local = local / local.max()
                result[rows] = candidate_weight * (local / local.sum())
    result /= result.sum()
    if not np.isfinite(result).all() or not (result > 0).all():
        raise ValueError("Invalid hierarchical training weights")
    return result


def _matrix(x):
    import numpy as np

    result = np.asarray(x, dtype=np.float64)
    if result.ndim != 2 or not np.isfinite(result).all():
        raise ValueError("Features must be a finite two-dimensional matrix")
    return result


def _standardize(x, mean, std, numeric_dims):
    """Apply stored training statistics, preserving the six one-hot columns."""
    import numpy as np

    result = np.array(x, dtype=np.float64, copy=True, order="C")
    result[:, :numeric_dims] = (result[:, :numeric_dims] - mean) / std
    if not np.isfinite(result).all():
        raise ValueError("Standardized features are not finite")
    return result


def _loss_grad(parameters, x, y, row_weights, classes):
    """Weighted log loss and its analytic gradient, with frozen L2 penalties."""
    import numpy as np
    from scipy.special import expit, logsumexp

    n_features = x.shape[1]
    if classes == 2:
        coef, intercept = parameters[:n_features], parameters[n_features]
        logits = x @ coef + intercept
        loss = np.dot(row_weights, np.logaddexp(0.0, logits) - y * logits)
        residual = row_weights * (expit(logits) - y)
        gradient = np.empty_like(parameters)
        gradient[:n_features] = x.T @ residual + NONBIAS_L2 * coef
        gradient[n_features] = residual.sum() + BIAS_L2 * intercept
    else:
        coef = parameters[:n_features * classes].reshape(n_features, classes)
        intercept = parameters[n_features * classes:]
        logits = x @ coef + intercept
        log_prob = logits - logsumexp(logits, axis=1, keepdims=True)
        loss = -np.dot(row_weights, log_prob[np.arange(len(y)), y])
        residual = np.exp(log_prob)
        residual[np.arange(len(y)), y] -= 1.0
        residual *= row_weights[:, None]
        gradient = np.concatenate((
            (x.T @ residual + NONBIAS_L2 * coef).ravel(),
            residual.sum(axis=0) + BIAS_L2 * intercept,
        ))
    loss += 0.5 * NONBIAS_L2 * np.square(coef).sum()
    loss += 0.5 * BIAS_L2 * np.square(intercept).sum()
    if not np.isfinite(loss) or not np.isfinite(gradient).all():
        raise FloatingPointError("Nonfinite frozen objective or gradient")
    return float(loss), gradient


def fit(x, y, weights, numeric_dims, classes):
    """Fit one fixed logistic/softmax model using only supplied training rows.

    Supported layouts are region: 24 features/18 numeric/2 classes, and whole:
    28 features/22 numeric/4 classes. ``mean`` and ``std`` contain numeric columns
    only; ``coef`` includes all feature columns. Binary probabilities refer to
    label 1; softmax class order is A, B, union, intersection (labels 0..3).

    Exactly one zero-initialized SciPy L-BFGS-B call is made. Unsuccessful
    termination raises RuntimeError; no retry or parameter adjustment occurs.
    """
    import numpy as np
    from scipy.optimize import minimize

    x = _matrix(x)
    if classes not in _LAYOUTS:
        raise ValueError("The frozen tasks have exactly 2 or 4 classes")
    n_features, expected_numeric = _LAYOUTS[classes]
    if x.shape[0] == 0 or x.shape[1] != n_features or numeric_dims != expected_numeric:
        raise ValueError("Feature dimensions differ from the frozen task layout")
    labels = np.asarray(y, dtype=np.float64)
    if labels.shape != (len(x),) or not np.isfinite(labels).all():
        raise ValueError("Labels must be a finite vector with one value per row")
    if not ((labels >= 0) & (labels < classes) & (labels == np.floor(labels))).all():
        raise ValueError("Labels must be integer class indices")
    labels = labels.astype(np.int64)
    row_weights = np.asarray(weights, dtype=np.float64)
    if row_weights.shape != (len(x),) or not np.isfinite(row_weights).all():
        raise ValueError("Training weights must be finite and one per row")
    if not (row_weights > 0).all() or not np.isclose(
        row_weights.sum(), 1.0, rtol=1e-12, atol=1e-12
    ):
        raise ValueError("Training weights must be strictly positive and sum to one")
    row_weights = row_weights / row_weights.sum()

    numeric = x[:, :numeric_dims]
    mean = np.sum(row_weights[:, None] * numeric, axis=0)
    # A mathematically constant column has exactly zero standard deviation.
    constant = np.all(numeric == numeric[0], axis=0)
    mean[constant] = numeric[0, constant]
    std = np.sqrt(np.sum(row_weights[:, None] * np.square(numeric - mean), axis=0))
    std[std == 0.0] = 1.0
    if not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise ValueError("Nonfinite training mean or standard deviation")
    standardized = _standardize(x, mean, std, numeric_dims)

    n_parameters = (n_features + 1) * (1 if classes == 2 else classes)
    initial = np.zeros(n_parameters, dtype=np.float64)
    initial_loss, _ = _loss_grad(initial, standardized, labels, row_weights, classes)
    result = minimize(
        _loss_grad,
        initial,
        args=(standardized, labels, row_weights, classes),
        method="L-BFGS-B",
        jac=True,
        options=dict(OPTIMIZER_OPTIONS),
    )
    if not result.success:
        raise RuntimeError(
            "Frozen L-BFGS-B did not converge: status=%s, iterations=%s, message=%s"
            % (result.status, result.nit, result.message)
        )
    final_loss, final_gradient = _loss_grad(
        result.x, standardized, labels, row_weights, classes
    )
    if not np.isfinite(result.x).all():
        raise RuntimeError("Frozen L-BFGS-B returned nonfinite parameters")
    if classes == 2:
        coef = result.x[:n_features].tolist()
        intercept = float(result.x[n_features])
    else:
        coef = result.x[:n_features * classes].reshape(n_features, classes).tolist()
        intercept = result.x[n_features * classes:].tolist()
    model = {
        "schema_version": 1,
        "type": "binary_logistic" if classes == 2 else "multinomial_softmax",
        "classes": int(classes),
        "class_order": [0, 1] if classes == 2 else ["A", "B", "union", "intersection"],
        "numeric_dims": int(numeric_dims),
        "n_features": int(n_features),
        "mean": mean.tolist(),
        "std": std.tolist(),
        "coef": coef,
        "intercept": intercept,
        "regularization": {"nonbias_l2": NONBIAS_L2, "bias_l2": BIAS_L2},
        "optimizer": {
            "method": "L-BFGS-B",
            "dtype": "float64",
            "initialization": "all_zero",
            "options": dict(OPTIMIZER_OPTIONS),
            "success": bool(result.success),
            "status": int(result.status),
            "message": str(result.message),
            "nit": int(result.nit),
            "nfev": int(result.nfev),
            "njev": int(result.njev),
            "initial_loss": initial_loss,
            "final_loss": final_loss,
            "gradient_inf_norm": float(np.abs(final_gradient).max()),
        },
        "training": {
            "rows": int(len(x)),
            "weight_sum": float(row_weights.sum()),
            "class_counts": np.bincount(labels, minlength=classes).tolist(),
            "class_weights": np.bincount(
                labels, weights=row_weights, minlength=classes
            ).tolist(),
            "zero_std_columns": np.flatnonzero(constant).tolist(),
        },
    }
    # Disallow JSON NaN/Infinity even if a future edit adds diagnostic fields.
    json.dumps(model, allow_nan=False)
    return model


def predict(model, x):
    """Return label-1 probabilities [N] or frozen-order softmax probabilities [N,4]."""
    import numpy as np
    from scipy.special import expit, softmax

    x = _matrix(x)
    classes = model["classes"]
    if classes not in _LAYOUTS:
        raise ValueError("Unsupported model class count")
    n_features, numeric_dims = _LAYOUTS[classes]
    if (
        model["n_features"] != n_features
        or model["numeric_dims"] != numeric_dims
        or x.shape[1] != n_features
    ):
        raise ValueError("Prediction layout differs from the frozen model")
    mean, std = [np.asarray(model[key], dtype=np.float64) for key in ("mean", "std")]
    if (
        mean.shape != (numeric_dims,)
        or std.shape != (numeric_dims,)
        or not np.isfinite(mean).all()
        or not np.isfinite(std).all()
        or not (std > 0).all()
    ):
        raise ValueError("Invalid stored training statistics")
    standardized = _standardize(x, mean, std, numeric_dims)
    coef = np.asarray(model["coef"], dtype=np.float64)
    intercept = np.asarray(model["intercept"], dtype=np.float64)
    coef_shape = (n_features,) if classes == 2 else (n_features, classes)
    intercept_shape = () if classes == 2 else (classes,)
    if coef.shape != coef_shape or intercept.shape != intercept_shape:
        raise ValueError("Invalid stored coefficient shape")
    if not np.isfinite(coef).all() or not np.isfinite(intercept).all():
        raise ValueError("Invalid stored coefficients")
    logits = standardized @ coef + intercept
    if not np.isfinite(logits).all():
        raise ValueError("Nonfinite prediction logits")
    return expit(logits) if classes == 2 else softmax(logits, axis=1)


def self_test():
    """Synthetic-only checks; one binary and one softmax optimizer call in total."""
    import numpy as np

    rng = np.random.default_rng(20261103)

    # Deliberately unequal row counts and component areas expose row weighting.
    images = [10, 10, 10, 10, 20, 20, 20]
    candidates = [0, 0, 0, 1, 0, 1, 1]
    w_region = weights(images, candidates, [1, 2, 3, 20, 7, 2, 6])
    expected = np.array([1/24, 1/12, 1/8, 1/4, 1/4, 1/16, 3/16])
    np.testing.assert_allclose(w_region, expected, rtol=0, atol=1e-15)
    np.testing.assert_allclose(weights([10, 10, 20], [0, 1, 0]), [.25, .25, .5])
    for image in set(images):
        np.testing.assert_allclose(w_region[np.array(images) == image].sum(), .5)
    rejected_duplicates = False
    try:
        weights([10, 10], [0, 0])
    except ValueError:
        rejected_duplicates = True
    assert rejected_duplicates, "Whole task accepted multiple rows per candidate"

    image_manifest = list(range(101, 142))
    folds = make_folds(image_manifest)
    assert folds == make_folds(list(reversed(image_manifest)) + image_manifest[:3])
    assert set(folds) == set(image_manifest) and set(folds.values()) == set(range(5))
    counts = np.bincount(list(folds.values()), minlength=5)
    assert counts.max() - counts.min() <= 1
    # Include both families, multiple candidates, categories and component rows.
    rows = [(image, family, category, candidate, region)
            for image in image_manifest for family in ("m2f", "rcnn")
            for category in (1, 6) for candidate in (0, 1) for region in (0, 1, 2)]
    for fold in range(5):
        train = {row[0] for row in rows if folds[row[0]] != fold}
        heldout = {row[0] for row in rows if folds[row[0]] == fold}
        assert train.isdisjoint(heldout) and train | heldout == set(image_manifest)

    gradient_errors = {}
    for classes in (2, 4):
        small_x = rng.normal(size=(9, 5))
        small_y = np.arange(9) % classes
        small_w = np.arange(1.0, 10.0); small_w /= small_w.sum()
        count = 6 * (1 if classes == 2 else classes)
        parameters = rng.normal(scale=.3, size=count)
        _, analytic = _loss_grad(parameters, small_x, small_y, small_w, classes)
        numerical = np.zeros_like(parameters)
        for index in range(count):
            plus, minus = parameters.copy(), parameters.copy()
            plus[index] += 1e-6; minus[index] -= 1e-6
            numerical[index] = (
                _loss_grad(plus, small_x, small_y, small_w, classes)[0]
                - _loss_grad(minus, small_x, small_y, small_w, classes)[0]
            ) / 2e-6
        np.testing.assert_allclose(analytic, numerical, rtol=2e-6, atol=2e-8)
        gradient_errors[str(classes)] = float(np.abs(analytic - numerical).max())

    model_checks = {}
    for classes in (2, 4):
        n_features, numeric_dims = _LAYOUTS[classes]
        n = 60 if classes == 2 else 64
        x = rng.normal(size=(n, n_features))
        x[:, numeric_dims-1] = 7.0
        x[:, numeric_dims:] = np.eye(6)[np.arange(n) % 6]
        if classes == 2:
            training_weights = weights(
                [100 + index//12 for index in range(n)],
                [index//3 for index in range(n)],
                1.0 + np.arange(n) % 3,
            )
            y = (x[:, 0] + .6*x[:, 1] > 0).astype(int)
        else:
            training_weights = weights(
                [100 + index//8 for index in range(n)], list(range(n))
            )
            y = np.arange(n) % 4
            x[:, 0] += 2 * (y - 1.5)
            x[:, 1] += 2 * np.array([1., -1., -1., 1.])[y]
        original = x.copy()
        # These are the ONLY two synthetic optimization calls in this test.
        model = fit(x, y, training_weights, numeric_dims, classes)
        np.testing.assert_array_equal(x, original)
        expected_mean = np.average(x[:, :numeric_dims], axis=0, weights=training_weights)
        expected_std = np.sqrt(np.average(
            (x[:, :numeric_dims] - expected_mean)**2,
            axis=0, weights=training_weights,
        ))
        expected_std[numeric_dims-1] = 1.0
        np.testing.assert_allclose(model["mean"], expected_mean, rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(model["std"], expected_std, rtol=1e-13, atol=1e-13)
        assert model["std"][numeric_dims-1] == 1.0
        transformed = _standardize(
            x, np.asarray(model["mean"]), np.asarray(model["std"]), numeric_dims
        )
        np.testing.assert_array_equal(transformed[:, numeric_dims:], x[:, numeric_dims:])
        stored = json.dumps(model, sort_keys=True, allow_nan=False)
        heldout = np.full((2, n_features), 1e4)
        heldout[:, numeric_dims:] = np.eye(6)[:2]
        mixed_mean = np.vstack((x, heldout))[:, :numeric_dims].mean(axis=0)
        assert not np.allclose(model["mean"], mixed_mean)
        probability = predict(model, x)
        # A changed held-out batch cannot recompute or mutate training statistics.
        combined = predict(model, np.vstack((x[:1], heldout)))
        np.testing.assert_allclose(combined[0], probability[0], rtol=1e-13, atol=1e-13)
        assert json.dumps(model, sort_keys=True, allow_nan=False) == stored
        restored = json.loads(stored)
        np.testing.assert_array_equal(predict(restored, x), probability)
        assert np.isfinite(probability).all()
        assert ((probability >= 0) & (probability <= 1)).all()
        assert probability.shape == ((n,) if classes == 2 else (n, 4))
        if classes == 4:
            np.testing.assert_allclose(probability.sum(axis=1), 1.0, atol=1e-14)
        model_checks[str(classes)] = {
            "rows": n,
            "optimizer_success": model["optimizer"]["success"],
            "iterations": model["optimizer"]["nit"],
            "final_loss": model["optimizer"]["final_loss"],
            "training_statistics_only": True,
            "json_roundtrip_exact": True,
        }
    return {
        "status": "passed",
        "synthetic_only": True,
        "optimizer_calls": 2,
        "weights_hierarchy": True,
        "shared_image_folds_no_leak": True,
        "finite_difference_max_errors": gradient_errors,
        "models": model_checks,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", help="Run synthetic-only checks")
    arguments = parser.parse_args()
    if not arguments.self_test:
        parser.error("Only --self-test is supported; real data are handled by the runner")
    print(json.dumps(self_test(), sort_keys=True, indent=2, allow_nan=False))
