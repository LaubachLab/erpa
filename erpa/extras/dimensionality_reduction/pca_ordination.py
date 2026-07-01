"""
PCA ordination of ERPA measure tables.

This module applies PCA to per-trial scalar measures and optionally applies a
varimax rotation to retained components. The analysis summarizes how trials vary
across the measure space.
"""

import numpy as np

from erpa.spatiotemporal.measures import measure_columns


def _varimax(L, max_iter=100, tol=1e-6):
    """
    Rotate a loading matrix with varimax.

    Parameters
    ----------
    L : numpy.ndarray
        Loading matrix with shape ``(n_measures, n_components)``.
    max_iter : int, optional
        Maximum number of rotation iterations.
    tol : float, optional
        Relative convergence tolerance.

    Returns
    -------
    L_rot : numpy.ndarray
        Varimax-rotated loading matrix.
    R : numpy.ndarray
        Orthogonal rotation matrix.

    Notes
    -----
    This implementation uses orthogonal varimax rotation without Kaiser
    normalization.
    """
    L = np.asarray(L, dtype=float)
    p, k = L.shape
    if k < 2:
        return L.copy(), np.eye(k)
    R = np.eye(k)
    d = 0.0
    for _ in range(max_iter):
        d_old = d
        Lr = L @ R
        u, s, vt = np.linalg.svd(
            L.T @ (Lr ** 3 - (1.0 / p) * Lr @ np.diag(np.diag(Lr.T @ Lr)))
        )
        R = u @ vt
        d = float(np.sum(s))
        if d_old != 0.0 and d / d_old < 1.0 + tol:
            break
    return L @ R, R


def pca_ordination(table, measures=None, n_components=4, standardize=True):
    """
    Compute PCA ordination for selected measure columns.

    Rows with missing values in selected measures are dropped before PCA.
    Standardization is applied by default so that measures with larger numerical
    scales do not dominate the components.

    Parameters
    ----------
    table : pandas.DataFrame
        Per-trial measure table from ``build_measure_table``.
    measures : list of str or None, optional
        Measure columns to include. If ``None``, all numeric measure columns
        returned by ``measure_columns`` are used.
    n_components : int, optional
        Number of principal components returned in the unrotated PCA output.
    standardize : bool, optional
        If ``True``, z-score each selected measure before PCA.

    Returns
    -------
    dict
        Dictionary with PCA and varimax outputs. Keys include ``"scores"``,
        ``"loadings"``, ``"eigenvalues"``, ``"explained_variance"``,
        ``"varimax_loadings"``, ``"varimax_scores"``, ``"n_rotated"``,
        ``"measures"``, and ``"rows"``.

    Notes
    -----
    The number of rotated components is selected with the Kaiser rule
    (eigenvalue greater than 1), with at least one component retained. Varimax
    rotation is added to the output; it does not replace the unrotated PCA scores
    or loadings.
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    cols = measures if measures is not None else measure_columns(table)
    sub = table.dropna(subset=cols)
    X = sub[cols].to_numpy(dtype=float)
    if standardize:
        X = StandardScaler().fit_transform(X)

    # Fit the full spectrum so the Kaiser rule sees every eigenvalue.
    full = min(X.shape[1], max(1, X.shape[0] - 1))
    pca = PCA(n_components=full)
    scores = pca.fit_transform(X)
    eig = pca.explained_variance_                          # eigenvalues
    evr = pca.explained_variance_ratio_
    k_show = min(n_components, full)

    # Varimax rotation of the Kaiser-retained components. This is additive. The
    # unrotated PCA outputs are returned unchanged alongside it. Rotation pushes
    # each measure toward one component, which aids reading the axes.
    k_rot = int(max(1, int((eig > 1).sum())))
    L = pca.components_[:k_rot].T * np.sqrt(eig[:k_rot])    # (n_measures, k_rot)
    L_rot, R = _varimax(L)
    Z = scores[:, :k_rot] / np.sqrt(eig[:k_rot])            # unit-variance scores
    Z_rot = Z @ R

    return dict(
        scores=scores[:, :k_show],
        loadings=pca.components_[:k_show],
        eigenvalues=eig,
        explained_variance=evr[:k_show],
        varimax_loadings=L_rot,
        varimax_scores=Z_rot,
        n_rotated=k_rot,
        measures=list(cols),
        rows=sub["idx"].to_numpy() if "idx" in sub else np.arange(len(sub)),
    )
