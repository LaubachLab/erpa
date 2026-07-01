#!/usr/bin/env python
# coding: utf-8
"""Variability summaries for decision trajectories.

This module summarizes one-dimensional deviation curves with functional PCA and
summarizes two-dimensional movement paths with Fisher-Rao elastic shape analysis.
It also includes a KMeans silhouette check for discrete clusters in score space.
"""

from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import warnings

import numpy as np
from numpy.typing import ArrayLike
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr

from erpa.util import as_meta_frame
from erpa.spatiotemporal.spatial import (decision_axis, arclength_resample,
                                       _segment_xy)


# ======================================================================
# fPCA variability of the deviation curves
# ======================================================================

def fpca_summary(
    grid: ArrayLike,
    Y: ArrayLike,
    meta: pd.DataFrame,
    n_components: int = 4,
    n_sd: float = 1.5,
    outcome: str = "error",
    cue: str = "cue",
    ylabel: str = "deviation",
) -> Dict[str, Any]:
    """Summarize curve variability with functional PCA.

    The input curves are centered, decomposed with singular value
    decomposition, and summarized as principal modes of variation. Each
    component score is compared by outcome and correlated with cue level.

    Parameters
    ----------
    grid : array-like of shape (n_points,)
        Grid for the curve samples. This argument is accepted for consistency
        with plotting functions but is not used in the calculation.
    Y : array-like of shape (n_trials, n_points)
        Curve matrix to summarize.
    meta : pandas.DataFrame
        Trial metadata with columns named by ``outcome`` and ``cue``.
    n_components : int, optional
        Maximum number of functional principal components to return.
    n_sd : float, optional
        Score standard deviation multiplier used to build the positive and
        negative mode curves.
    outcome : str, optional
        Metadata column used for the binary outcome contrast. Values of 1 are
        treated as correct trials and values of 0 are treated as error trials.
    cue : str, optional
        Metadata column used for the cue correlation.
    ylabel : str, optional
        Label accepted for compatibility with plotting workflows. This argument
        is not used in the calculation.

    Returns
    -------
    dict
        Dictionary with the following keys:

        ``mean`` : ndarray of shape (n_points,)
            Mean curve.
        ``components`` : ndarray of shape (n_components, n_points)
            Functional principal component curves.
        ``scores`` : ndarray of shape (n_trials, n_components)
            Per-trial component scores.
        ``var_ratio`` : ndarray of shape (n_components,)
            Fraction of variance explained by each component.
        ``modes`` : list of tuple of ndarray
            Positive and negative mode curves for plotting.
        ``table`` : pandas.DataFrame
            Per-component outcome contrasts and cue correlations.

    Notes
    -----
    Component signs are arbitrary. Interpret score magnitudes and relative
    contrasts rather than the sign alone.
    """
    Y = np.asarray(Y, float)
    mean = Y.mean(0)
    Yc = Y - mean
    U, S, Vt = np.linalg.svd(Yc, full_matrices=False)
    k = min(n_components, Vt.shape[0])
    comp = Vt[:k]                       # (k, n_points) eigenfunctions
    scores = Yc @ comp.T               # (n, k)
    var_ratio = (S ** 2 / np.sum(S ** 2))[:k]

    error = np.asarray(meta[outcome].values)
    cue_v = np.asarray(meta[cue].values)
    rows = []
    for j in range(k):
        sj = scores[:, j]
        h, m = sj[error == 0], sj[error == 1]
        try:
            up = mannwhitneyu(h, m).pvalue
        except ValueError:
            up = np.nan
        rho, rp = spearmanr(cue_v, sj, nan_policy="omit")
        rows.append(dict(PC=j + 1, var=float(var_ratio[j]),
                         cor_minus_err=float(np.nanmean(h) - np.nanmean(m)),
                         outcome_p=float(up), cue_rho=float(rho),
                         cue_p=float(rp)))
    table = pd.DataFrame(rows)

    modes = [(mean + n_sd * scores[:, j].std() * comp[j],
              mean - n_sd * scores[:, j].std() * comp[j]) for j in range(k)]

    return dict(mean=mean, components=comp, scores=scores,
                var_ratio=var_ratio, modes=modes, table=table)


# ======================================================================
# Elastic shape analysis of the 2D paths
# ======================================================================

def decision_frame_paths(
    trials: Sequence[Dict[str, Any]],
    ports: Mapping[str, np.ndarray],
    n_points: int = 40,
    node: str = "centroid",
    node_names: Optional[Sequence[str]] = None,
    pix: float = 1.0,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Convert choice paths to a common decision frame.

    Each path is taken from center exit to choice entry, resampled at equal arc
    length, centered on the center port, and rotated into the decision frame.
    The chosen direction is positive on the y-axis. Left and right choices are
    folded into the same coordinate frame.

    Parameters
    ----------
    trials : list of dict
        Trial dictionaries from ``build_trials`` or ``load_session``.
    ports : dict
        Port locations from ``locate_ports``. The dictionary must include the
        port labels required by ``decision_axis``.
    n_points : int, optional
        Number of arc-length samples in each output path.
    node : str, optional
        Trial field or node name used to extract the path.
    node_names : sequence of str or None, optional
        Node names used when ``node`` selects a tracked pose node.
    pix : float, optional
        Multiplicative scale factor applied to output coordinates. Use a
        centimeters-per-pixel value to return paths in centimeters.

    Returns
    -------
    paths : ndarray of shape (n_valid_trials, n_points, 2)
        Side-folded paths in the decision frame.
    meta : pandas.DataFrame
        Metadata for the returned paths. Rows align with ``paths``.

    Notes
    -----
    Trials are skipped when the movement segment is invalid, too short, or
    contains non-finite coordinates.
    """
    center, u, dR, dL = decision_axis(ports)
    u_perp = np.array([-u[1], u[0]])
    paths, meta, n_skip = [], [], 0
    for t in trials:
        ce = t["events"]["center_exit"]
        ch = t["events"]["choice_entry"]
        if ce < 1 or ch >= len(t["lin_vel"]) or ch <= ce:
            n_skip += 1
            continue
        xy = _segment_xy(t, ce, ch, node, node_names)
        if xy is None or xy.shape[0] < 4 or not np.all(np.isfinite(xy)):
            n_skip += 1
            continue
        rs = arclength_resample(xy, n_points)          # (n_points, 2), pixels
        p = rs - center
        s = 1.0 if int(t["choice"]) == 1 else -1.0
        X = (p @ u_perp) * pix
        Yc = s * (p @ u) * pix
        paths.append(np.column_stack([X, Yc]))
        meta.append({kk: t[kk] for kk in ("idx", "target", "choice",
                                          "cue", "rt", "error") if kk in t})
    if n_skip:
        warnings.warn(f"decision_frame_paths skipped {n_skip} trials.")
    return np.array(paths), as_meta_frame(meta)


def elastic_shape_summary(
    trials: Sequence[Dict[str, Any]],
    ports: Mapping[str, np.ndarray],
    n_points: int = 40,
    node: str = "centroid",
    node_names: Optional[Sequence[str]] = None,
    pix: float = 1.0,
    n_components: int = 4,
    scale: bool = False,
    outcome: str = "error",
    cue: str = "cue",
) -> Dict[str, Any]:
    """Summarize two-dimensional choice paths with elastic shape analysis.

    Paths are folded into the decision frame, converted to an ``fdacurve``
    object, aligned by square-root velocity functions, and decomposed with shape
    PCA. Each shape score is compared by outcome and correlated with cue level.

    Parameters
    ----------
    trials : list of dict
        Trial dictionaries from ``build_trials`` or ``load_session``.
    ports : dict
        Port locations from ``locate_ports``.
    n_points : int, optional
        Number of arc-length samples in each path.
    node : str, optional
        Trial field or node name used to extract the path.
    node_names : sequence of str or None, optional
        Node names used when ``node`` selects a tracked pose node.
    pix : float, optional
        Multiplicative scale factor applied to output coordinates before shape
        analysis. Use a centimeters-per-pixel value for centimeter units.
    n_components : int, optional
        Number of principal shape components to compute.
    scale : bool, optional
        If ``False``, physical size is retained. If ``True``, shape analysis is
        scale-normalized.
    outcome : str, optional
        Metadata column used for the binary outcome contrast. Values of 1 are
        treated as errors and values of 0 are treated as correct.
    cue : str, optional
        Metadata column used for the cue correlation.

    Returns
    -------
    dict
        Dictionary with the following keys:

        ``mean_shape`` : ndarray of shape (n_points, 2)
            Karcher mean shape.
        ``scores`` : ndarray of shape (n_trials, n_components)
            Per-trial shape PCA scores.
        ``var_ratio`` : ndarray
            Variance fraction for each shape component.
        ``modes`` : ndarray
            Shape modes returned by ``fdacurve.shape_pca``.
        ``table`` : pandas.DataFrame
            Per-component outcome contrasts and cue correlations.
        ``meta`` : pandas.DataFrame
            Metadata for the paths included in the analysis.

    Notes
    -----
    This function requires ``fdasrsf``.
    """
    from fdasrsf import fdacurve
    paths, meta = decision_frame_paths(trials, ports, n_points, node,
                                       node_names, pix)
    beta = np.stack([p.T for p in paths], axis=2)      # (2, n_points, K)
    obj = fdacurve(beta, mode="O", N=n_points, scale=scale)
    obj.karcher_mean()
    obj.srvf_align()
    obj.shape_pca(no=n_components)

    scores = np.asarray(obj.coef).T                    # (K, n_components)
    s = np.asarray(obj.s, float)
    var_ratio = s / s.sum()
    mean_shape = np.asarray(obj.beta_mean).T           # (n_points, 2)
    modes = np.asarray(obj.pca)                        # (2, n_points, nc, steps)

    error = np.asarray(meta[outcome].values)
    cue_v = np.asarray(meta[cue].values)
    rows = []
    for j in range(min(n_components, scores.shape[1])):
        sj = scores[:, j]
        h, m = sj[error == 0], sj[error == 1]
        try:
            up = mannwhitneyu(h, m).pvalue
        except ValueError:
            up = np.nan
        rho, rp = spearmanr(cue_v, sj, nan_policy="omit")
        rows.append(dict(PC=j + 1, var=float(var_ratio[j]),
                         cor_minus_err=float(np.nanmean(h) - np.nanmean(m)),
                         outcome_p=float(up), cue_rho=float(rho),
                         cue_p=float(rp)))
    table = pd.DataFrame(rows)

    return dict(mean_shape=mean_shape, scores=scores, var_ratio=var_ratio,
                modes=modes, table=table, meta=meta)


# ======================================================================
# KMeans discreteness check
# ======================================================================

def cluster_check(
    scores: ArrayLike,
    ks: Sequence[int] = (2, 3, 4),
    min_frac: float = 0.05,
) -> pd.DataFrame:
    """Check KMeans cluster solutions with silhouette scores.

    Each value of ``k`` is fit with KMeans. The output reports the silhouette
    score and the smallest cluster fraction. A solution is flagged when the
    smallest cluster contains less than ``min_frac`` of the observations.

    Parameters
    ----------
    scores : array-like of shape (n_trials, n_features)
        Score matrix used for clustering.
    ks : sequence of int, optional
        Numbers of clusters to test.
    min_frac : float, optional
        Minimum acceptable fraction of observations in the smallest cluster.

    Returns
    -------
    pandas.DataFrame
        Table with columns ``k``, ``silhouette``, ``min_cluster_frac``, and
        ``flag``.

    Notes
    -----
    A high silhouette score can occur when a small outlier group is separated
    from the main distribution. Check ``min_cluster_frac`` before interpreting a
    cluster solution as discrete structure.
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    scores = np.asarray(scores)
    out = []
    for k in ks:
        lab = KMeans(k, n_init=10, random_state=0).fit_predict(scores)
        frac = np.bincount(lab).min() / len(lab)
        out.append(dict(k=k, silhouette=round(float(silhouette_score(scores, lab)), 3),
                        min_cluster_frac=round(float(frac), 3),
                        flag="outlier-split" if frac < min_frac else "ok"))
    return pd.DataFrame(out)
