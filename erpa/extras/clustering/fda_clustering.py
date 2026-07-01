"""
Clustering of registered ERPA curves.

This module clusters curves or FDA score matrices stored in an xarray Dataset
created by ``erpa.fda.dataset.register_to_dataset``. It supports K-medoids
clustering with Euclidean or elastic time-series distances.
"""

import numpy as np
import pandas as pd

try:
    from aeon.clustering import TimeSeriesKMedoids
    from aeon.distances import pairwise_distance as aeon_pairwise
    HAVE_AEON = True
except Exception:
    HAVE_AEON = False


# ======================================================================
# KMedoids on a precomputed distance matrix (fallback when aeon is absent)
# ======================================================================

def _kmedoids_once(D, k, rng, max_iter):
    """
    Run one K-medoids fit on a precomputed distance matrix.

    Parameters
    ----------
    D : numpy.ndarray
        Distance matrix with shape ``(n_samples, n_samples)``.
    k : int
        Number of clusters.
    rng : numpy.random.Generator
        Random number generator used for initialization.
    max_iter : int
        Maximum number of update iterations.

    Returns
    -------
    labels : numpy.ndarray
        Cluster label for each sample.
    medoids : numpy.ndarray
        Row indices of the medoid samples.
    inertia : float
        Sum of distances from samples to their assigned medoids.
    """
    n = D.shape[0]
    # k-medoids++ seeding.
    medoids = [int(rng.integers(n))]
    closest = D[medoids[0]].copy()
    for _ in range(1, k):
        w = closest ** 2
        s = w.sum()
        cand = int(rng.integers(n)) if s == 0 else int(rng.choice(n, p=w / s))
        medoids.append(cand)
        closest = np.minimum(closest, D[cand])
    medoids = np.array(sorted(set(medoids)))
    while medoids.size < k:                       # repair collapsed seeds
        cand = int(rng.integers(n))
        if cand not in medoids:
            medoids = np.append(medoids, cand)

    for _ in range(max_iter):
        labels = np.argmin(D[:, medoids], axis=1)
        new = medoids.copy()
        for j in range(k):
            members = np.where(labels == j)[0]
            if members.size:
                within = D[np.ix_(members, members)].sum(axis=1)
                new[j] = members[int(np.argmin(within))]
        if np.array_equal(np.sort(new), np.sort(medoids)):
            medoids = new
            break
        medoids = new

    labels = np.argmin(D[:, medoids], axis=1)
    inertia = float(D[np.arange(n), medoids[labels]].sum())
    return labels, medoids, inertia


def kmedoids(D, k, n_init=10, max_iter=300, random_state=0):
    """
    Cluster a precomputed distance matrix with K-medoids.

    Parameters
    ----------
    D : numpy.ndarray
        Symmetric distance matrix with shape ``(n_samples, n_samples)``.
    k : int
        Number of clusters.
    n_init : int, optional
        Number of random initializations.
    max_iter : int, optional
        Maximum number of update iterations per initialization.
    random_state : int, optional
        Random seed.

    Returns
    -------
    labels : numpy.ndarray
        Cluster label for each sample.
    medoids : numpy.ndarray
        Row indices of the medoid samples.
    inertia : float
        Sum of distances from samples to their assigned medoids.
    """
    rng = np.random.default_rng(random_state)
    best = None
    for _ in range(n_init):
        out = _kmedoids_once(D, k, rng, max_iter)
        if best is None or out[2] < best[2]:
            best = out
    return best


def _medoids_from_distance(D, labels, k):
    """
    Select one medoid index for each cluster from a distance matrix.

    Parameters
    ----------
    D : numpy.ndarray
        Distance matrix with shape ``(n_samples, n_samples)``.
    labels : numpy.ndarray
        Cluster labels.
    k : int
        Number of clusters.

    Returns
    -------
    numpy.ndarray
        Row index of the within-cluster distance minimizer for each cluster.
    """
    medoids = np.zeros(k, dtype=int)
    for j in range(k):
        members = np.where(labels == j)[0]
        if members.size == 0:
            medoids[j] = 0
            continue
        within = D[np.ix_(members, members)].sum(axis=1)
        medoids[j] = members[int(np.argmin(within))]
    return medoids


def _cluster_matrix(X, k, distance, method, random_state, n_init=10):
    """
    Cluster an array with K-medoids and return the distance matrix.

    Uses aeon when available. If aeon is unavailable, only Euclidean distance is
    supported through the local K-medoids fallback.

    Parameters
    ----------
    X : numpy.ndarray
        Data matrix. A two-dimensional array is treated as univariate
        time-series data by aeon.
    k : int
        Number of clusters.
    distance : str
        Distance metric name.
    method : str
        K-medoids method name used by aeon.
    random_state : int
        Random seed.
    n_init : int, optional
        Number of random initializations.

    Returns
    -------
    labels : numpy.ndarray
        Cluster labels.
    medoids : numpy.ndarray
        Medoid row indices.
    D : numpy.ndarray
        Pairwise distance matrix.

    Raises
    ------
    RuntimeError
        If aeon is unavailable and a non-Euclidean distance is requested.
    """
    if HAVE_AEON:
        Xa = X[:, None, :] if X.ndim == 2 else X
        km = TimeSeriesKMedoids(
            n_clusters=k, distance=distance, method=method,
            n_init=n_init, random_state=random_state,
        )
        labels = km.fit_predict(Xa)
        D = aeon_pairwise(Xa, method=distance)
        return labels, _medoids_from_distance(D, labels, k), D

    if distance != "euclidean":
        raise RuntimeError(
            f"aeon is not installed, so distance '{distance}' is unavailable. "
            f"Install aeon for elastic distances, or use 'euclidean'."
        )
    from sklearn.metrics import pairwise_distances

    D = pairwise_distances(X, metric="euclidean")
    labels, medoids, _ = kmedoids(D, k, n_init=n_init, random_state=random_state)
    return labels, medoids, D


def cluster_curves(
    ds, source="registered", k=3, distance="euclidean", method="pam",
    random_state=0, condition_vars=("target", "cue"),
):
    """
    Cluster one Dataset variable and summarize cluster composition.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset from ``register_to_dataset``.
    source : str, optional
        Dataset variable to cluster. Common choices are ``"registered"``,
        ``"warping"``, ``"raw"``, ``"amp_scores"``, ``"phase_scores"``, and
        ``"joint_scores"``.
    k : int, optional
        Number of clusters.
    distance : str, optional
        Distance metric. Use ``"euclidean"`` for registered curves and score
        matrices. Elastic distances such as ``"dtw"``, ``"msm"``, or ``"twe"``
        require aeon.
    method : str, optional
        K-medoids method passed to aeon.
    random_state : int, optional
        Random seed.
    condition_vars : tuple of str, optional
        Trial coordinates used for cluster-by-condition crosstabs.

    Returns
    -------
    labels : numpy.ndarray
        Cluster label for each trial.
    medoid_trials : numpy.ndarray
        ``trial_idx`` values for the medoid trials.
    info : dict
        Dictionary containing ``"silhouette"``, ``"medoid_trials"``, and one
        crosstab per available condition variable.

    Raises
    ------
    ValueError
        If ``source`` is not a two-dimensional Dataset variable.
    """
    X = ds[source].values
    if X.ndim != 2:
        raise ValueError(f"'{source}' must be 2-D, got shape {X.shape}.")
    labels, medoids, D = _cluster_matrix(
        X, k, distance, method, random_state
    )

    from sklearn.metrics import silhouette_score

    ds.coords["cluster"] = ("trial", labels)
    sil = (float(silhouette_score(D, labels, metric="precomputed"))
           if k > 1 and len(np.unique(labels)) > 1 else np.nan)

    info = {"silhouette": sil,
            "medoid_trials": ds["trial_idx"].values[medoids]}
    for cv in condition_vars:
        if cv in ds.coords:
            info[f"crosstab_{cv}"] = pd.crosstab(
                pd.Series(labels, name="cluster"),
                pd.Series(ds[cv].values, name=cv),
            )
    return labels, ds["trial_idx"].values[medoids], info


def cluster_timeseries(X, k=3, distance="dtw", method="pam", random_state=0):
    """
    Cluster a univariate or multivariate time-series array with aeon.

    Parameters
    ----------
    X : numpy.ndarray
        Time-series array with shape ``(n_cases, n_timepoints)`` or
        ``(n_cases, n_channels, n_timepoints)``.
    k : int, optional
        Number of clusters.
    distance : str, optional
        aeon distance metric.
    method : str, optional
        K-medoids method passed to aeon.
    random_state : int, optional
        Random seed.

    Returns
    -------
    labels : numpy.ndarray
        Cluster labels.
    medoid_rows : numpy.ndarray
        Row indices of medoid cases.
    silhouette : float
        Silhouette score computed from the pairwise distance matrix.

    Raises
    ------
    RuntimeError
        If aeon is not installed.
    """
    if not HAVE_AEON:
        raise RuntimeError("cluster_timeseries needs aeon installed.")
    Xa = X[:, None, :] if X.ndim == 2 else X
    km = TimeSeriesKMedoids(
        n_clusters=k, distance=distance, method=method, random_state=random_state
    )
    from sklearn.metrics import silhouette_score

    labels = km.fit_predict(Xa)
    D = aeon_pairwise(Xa, method=distance)
    sil = (float(silhouette_score(D, labels, metric="precomputed"))
           if len(np.unique(labels)) > 1 else np.nan)
    medoids = _medoids_from_distance(D, labels, k)
    return labels, medoids, sil


def suggest_k(ds, source="registered", distance="euclidean",
              method="pam", k_range=range(2, 7), random_state=0):
    """
    Compute silhouette scores across candidate cluster counts.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset from ``register_to_dataset``.
    source : str, optional
        Dataset variable to cluster.
    distance : str, optional
        Distance metric.
    method : str, optional
        K-medoids method passed to aeon.
    k_range : iterable of int, optional
        Candidate cluster counts.
    random_state : int, optional
        Random seed.

    Returns
    -------
    dict
        Mapping from cluster count to silhouette score.
    """
    from sklearn.metrics import silhouette_score

    X = ds[source].values
    out = {}
    for k in k_range:
        labels, _, D = _cluster_matrix(X, k, distance, method, random_state)
        out[int(k)] = (float(silhouette_score(D, labels, metric="precomputed"))
                       if len(np.unique(labels)) > 1 else np.nan)
    return out
