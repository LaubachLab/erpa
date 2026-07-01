"""
Tabular classification of ERPA measure tables.

This module builds a feature matrix from per-trial scalar measures and evaluates
a HistGradientBoosting classifier with SHAP-based feature importance. The helper
functions support held-out SHAP importance and stratified cross-validation.
"""

import numpy as np
import pandas as pd

from erpa.spatiotemporal.measures import build_measure_table, LABEL_COLUMNS


def build_feature_matrix(trials, ports, pix, target="target", node="centroid",
                         node_names=None, add_fda=True, key="lin_vel",
                         n_points=50, dropna=True):
    """
    Build a feature matrix and target vector for tabular classification.

    The function builds a measure table, selects numeric measure columns as
    features, and returns the requested behavioral label as the target. Columns
    listed in ``LABEL_COLUMNS`` are not used as features.

    Parameters
    ----------
    trials : list of dict
        Trial dictionaries from ``erpa.core.session.load_session`` or
        ``erpa.core.session.build_trials``.
    ports : dict
        Port locations used by ``build_measure_table``.
    pix : float or None
        Pixels-to-centimeters factor passed to ``build_measure_table``.
    target : str, optional
        Label column to predict, such as ``"target"``, ``"error"``, or
        ``"cue"``.
    node : str, optional
        Position source used for path measures.
    node_names : sequence of str or None, optional
        Pose-node names used when ``node`` selects a keypoint from the trial
        node array.
    add_fda : bool, optional
        If ``True``, include FDA-derived measures from ``build_measure_table``.
    key : str, optional
        Trial signal used to build FDA movement curves.
    n_points : int, optional
        Number of samples in each resampled FDA movement curve.
    dropna : bool, optional
        If ``True``, drop rows with missing feature or target values.

    Returns
    -------
    X : pandas.DataFrame
        Feature matrix with one row per kept trial.
    y : numpy.ndarray
        Target values aligned to ``X``.
    feature_names : list of str
        Feature column names.

    Raises
    ------
    KeyError
        If ``target`` is not present in the measure table.
    """
    table = build_measure_table(trials, ports, pix, node=node,
                                node_names=node_names, add_fda=add_fda,
                                key=key, n_points=n_points)
    if target not in table.columns:
        raise KeyError(f"target '{target}' is not a label column "
                       f"{list(LABEL_COLUMNS)}.")
    feature_names = [c for c in table.columns if c not in LABEL_COLUMNS]
    cols = feature_names + [target]
    sub = table.dropna(subset=cols) if dropna else table
    X = sub[feature_names].reset_index(drop=True)
    y = sub[target].to_numpy()
    return X, y, feature_names


def classifier_importance(X, y, n_estimators=200, test_size=0.3,
                          scoring="balanced_accuracy", random_state=0):
    """
    Fit a HistGradientBoosting classifier and estimate SHAP feature importance.

    The data are split into training and held-out test sets. The classifier is
    fit on the training set. SHAP values are computed on the held-out set using
    a TreeExplainer, and mean absolute SHAP values are returned as the feature
    importance summary.

    Parameters
    ----------
    X : pandas.DataFrame
        Feature matrix from ``build_feature_matrix``.
    y : array-like
        Target values aligned to ``X``.
    n_estimators : int, optional
        Maximum number of boosting iterations.
    test_size : float, optional
        Fraction of samples assigned to the held-out test set.
    scoring : str, optional
        scikit-learn scoring name used for the held-out score.
    random_state : int, optional
        Random seed for train-test splitting and model fitting.

    Returns
    -------
    dict
        Dictionary with keys ``"importance"``, ``"shap_values"``,
        ``"test_score"``, and ``"model"``. ``"importance"`` is a DataFrame
        sorted from highest to lowest mean absolute SHAP value. ``"shap_values"``
        is an array of shape ``(n_test, n_features)``.

    Notes
    -----
    ``HistGradientBoostingClassifier`` handles missing values natively, so rows
    with NaN features are not dropped internally. Pass a clean feature matrix
    from ``build_feature_matrix`` with ``dropna=True`` for standard use.
    SHAP and scikit-learn are imported inside this function.
    """
    import shap
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import get_scorer

    Xv = X.to_numpy(dtype=float)
    strat = y if len(np.unique(y)) > 1 else None
    Xtr, Xte, ytr, yte = train_test_split(
        Xv, y, test_size=test_size, random_state=random_state, stratify=strat)

    clf = HistGradientBoostingClassifier(
        max_iter=n_estimators, class_weight="balanced",
        random_state=random_state)
    clf.fit(Xtr, ytr)

    explainer = shap.TreeExplainer(clf)
    sv = explainer.shap_values(Xte)
    sv = np.asarray(sv)
    # TreeExplainer returns (n_samples, n_features) for binary classification
    # in recent shap versions; older versions may return (2, n_samples, n_features)
    if sv.ndim == 3:
        sv = sv[1]

    imp = pd.DataFrame({
        "feature": list(X.columns),
        "shap_importance": np.abs(sv).mean(axis=0),
        "shap_sd": np.abs(sv).std(axis=0),
    }).sort_values("shap_importance", ascending=False).reset_index(drop=True)

    score = get_scorer(scoring)(clf, Xte, yte)
    return dict(importance=imp, shap_values=sv, test_score=float(score), model=clf)


def cross_val_score(X, y, n_estimators=200, n_splits=5,
                    scoring="balanced_accuracy", random_state=0):
    """
    Compute stratified cross-validated HistGradientBoosting scores.

    Parameters
    ----------
    X : pandas.DataFrame
        Feature matrix.
    y : array-like
        Target values aligned to ``X``.
    n_estimators : int, optional
        Maximum number of boosting iterations.
    n_splits : int, optional
        Number of stratified cross-validation folds.
    scoring : str, optional
        scikit-learn scoring name.
    random_state : int, optional
        Random seed for cross-validation splitting and model fitting.

    Returns
    -------
    dict
        Dictionary with keys ``"scores"`` and ``"mean"``. ``"scores"`` contains
        one score per fold. ``"mean"`` is the mean score across folds.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import cross_val_score as _cvs
    from sklearn.model_selection import StratifiedKFold

    clf = HistGradientBoostingClassifier(
        max_iter=n_estimators, class_weight="balanced",
        random_state=random_state)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True,
                         random_state=random_state)
    scores = _cvs(clf, X.to_numpy(dtype=float), y, cv=cv, scoring=scoring)
    return dict(scores=scores, mean=float(np.mean(scores)))
