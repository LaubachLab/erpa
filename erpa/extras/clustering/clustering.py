"""
Density clustering of ERPA measure tables.

This module embeds per-trial scalar measures with PaCMAP and clusters the
embedding with HDBSCAN. The analysis groups trials by similarity in the measure
space without specifying the number of clusters.
"""

import numpy as np

from erpa.spatiotemporal.measures import measure_columns


def cluster_measures(table, measures=None, n_neighbors=None, n_components=2,
                     min_cluster_size=10, min_samples=None, standardize=True,
                     random_state=0):
    """
    Embed measure-table rows with PaCMAP and cluster them with HDBSCAN.

    Rows with missing values in the selected measures are dropped before
    embedding. Optional FDA-derived columns are included when they are present in
    the table and selected by ``measure_columns`` or by ``measures``.

    Parameters
    ----------
    table : pandas.DataFrame
        Per-trial measure table from ``build_measure_table``.
    measures : list of str or None, optional
        Measure columns used for embedding and clustering. If ``None``, all
        numeric measure columns returned by ``measure_columns`` are used.
    n_neighbors : int or None, optional
        Number of neighbors used by PaCMAP. If ``None``, PaCMAP selects a value
        based on the sample size.
    n_components : int, optional
        Number of PaCMAP embedding dimensions.
    min_cluster_size : int, optional
        Minimum number of samples in an HDBSCAN cluster.
    min_samples : int or None, optional
        HDBSCAN density-conservatism parameter. If ``None``, it is set to
        ``min_cluster_size``.
    standardize : bool, optional
        If ``True``, z-score each measure before embedding.
    random_state : int, optional
        Random seed passed to PaCMAP.

    Returns
    -------
    dict
        Dictionary with keys ``"embedding"``, ``"labels"``, ``"table"``,
        ``"measures"``, and ``"rows"``. ``"labels"`` contains HDBSCAN cluster
        labels, with ``-1`` marking noise points.

    Notes
    -----
    PaCMAP, fast_hdbscan, and scikit-learn are imported inside this function.
    Install the optional example dependencies before running this analysis.
    """
    import pacmap
    import fast_hdbscan
    from sklearn.preprocessing import StandardScaler

    cols = measures if measures is not None else measure_columns(table)
    sub = table.dropna(subset=cols).copy()
    X = sub[cols].to_numpy(dtype=float)
    if standardize:
        X = StandardScaler().fit_transform(X)

    emb = pacmap.PaCMAP(
        n_components=n_components, n_neighbors=n_neighbors,
        random_state=random_state,
    ).fit_transform(X)

    if min_samples is None:
        min_samples = min_cluster_size
    labels = fast_hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size, min_samples=min_samples,
    ).fit_predict(emb)

    for j in range(emb.shape[1]):
        sub[f"emb_{j + 1}"] = emb[:, j]
    sub["cluster"] = labels

    return dict(
        embedding=emb,
        labels=labels,
        table=sub,
        measures=list(cols),
        rows=sub["idx"].to_numpy() if "idx" in sub else np.arange(len(sub)),
    )


def plot_clusters(result, ax=None):
    """
    Plot a two-dimensional cluster embedding.

    Parameters
    ----------
    result : dict
        Output from ``cluster_measures``. Must contain ``"embedding"`` and
        ``"labels"``.
    ax : matplotlib.axes.Axes or None, optional
        Axes on which to draw the plot. If ``None``, a new figure and axes are
        created.

    Returns
    -------
    matplotlib.axes.Axes
        Axes containing the cluster plot.

    Raises
    ------
    ValueError
        If the embedding has fewer than two dimensions.

    Notes
    -----
    HDBSCAN label ``-1`` is plotted as noise.
    """
    import matplotlib.pyplot as plt

    emb = result["embedding"]
    labels = np.asarray(result["labels"])
    if emb.shape[1] < 2:
        raise ValueError("plot_clusters needs a 2D embedding.")
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    noise = labels == -1
    if noise.any():
        ax.scatter(emb[noise, 0], emb[noise, 1], s=12, c="0.7", label="noise")
    for c in sorted(set(labels[~noise].tolist())):
        m = labels == c
        ax.scatter(emb[m, 0], emb[m, 1], s=16, label=f"cluster {c}")
    ax.set_xlabel("PaCMAP 1")
    ax.set_ylabel("PaCMAP 2")
    ax.legend(loc="best", fontsize=8, frameon=False)
    return ax
