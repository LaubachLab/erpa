"""Correlation clustermap for ERPA scalar measure tables."""

import numpy as np
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr


def corr_clustermap(table, measures=None, prune=True, method="average",
                    n_groups=None, cmap="coolwarm", figsize=None):
    """Plot a Spearman correlation clustermap of scalar measures.

    Measures are reordered by hierarchical clustering on ``1 - |corr|``.
    Correlations with ``p > 0.05`` are masked to zero in the display.

    Parameters
    ----------
    table : pandas.DataFrame
        Output of ``build_measure_table``.
    measures : list of str or None, optional
        Columns to include. If ``None``, all non-label columns are used.
    prune : bool, optional
        If ``True``, drop the centimeter twin of a measure when its normalized
        twin is present. Pass ``measures`` to use an explicit set instead.
    method : str, optional
        Linkage method passed to ``scipy.cluster.hierarchy.linkage``.
    n_groups : int or None, optional
        If set, cut the tree into ``n_groups`` clusters, outline the diagonal
        blocks, and return a group label per measure in ``info["groups"]``.
    cmap : str, optional
        Colormap for the heatmap.
    figsize : tuple of float or None, optional
        Figure size in inches. If ``None``, size is derived from the number of
        measures.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure containing the dendrogram and heatmap.
    info : dict
        Dictionary with keys ``"order"``, ``"linkage"``, ``"measures"``, and
        ``"groups"``.

    Raises
    ------
    ValueError
        If fewer than three usable measures remain after filtering.

    Notes
    -----
    ``peak_ang_vel`` is converted to absolute values before correlation because
    it is a signed measure whose sign depends on turn direction.
    The label set and redundant pairs are tuned for the Laubach Lab task design.
    """
    # these values are specific to those used by the Laubach Lab
    # revise them as needed for your experimental design
    _LABELS = {"participant_id", "idx", "absolute_trial", "target", "choice", "hit",
               "cue_level", "RT", "treatment", "session_file"}
    _REDUNDANT_PAIRS = [("dev_unchosen_cm", "dev_unchosen_norm"),
                        ("path_len_cm", "path_len_norm")]

    # angular velocity should be set to absolute values, as this is a signed measure
    # i.e., trials with left and right movements have negative and positive signs
    table = table.copy()
    if "peak_ang_vel" in table.columns:
        table["peak_ang_vel"] = np.abs(table["peak_ang_vel"])

    cols = list(measures) if measures is not None else \
        [c for c in table.columns if c not in _LABELS]
    if prune:
        for cm, norm in _REDUNDANT_PAIRS:
            if cm in cols and norm in cols:
                cols.remove(cm)

    M = table[cols].dropna()
    cols = [c for c in cols if M[c].std() > 0]
    M = M[cols]
    n = len(cols)
    if n < 3:
        raise ValueError("need at least three usable measures for a clustermap")

    # Calculate Spearman correlation and p-values
    C, P = spearmanr(M)

    # Fallback if spearmanr returns scalars for small inputs
    if np.isscalar(C):
        C = np.array([[1.0, C], [C, 1.0]])
        P = np.array([[0.0, P], [P, 0.0]])

    np.fill_diagonal(C, 1.0)

    # Distance metric for clustering based on the unmasked correlations
    D = 1.0 - np.abs(C)
    np.fill_diagonal(D, 0.0)
    D = 0.5 * (D + D.T)                      # enforce symmetry for squareform

    # Hierarchical clustering
    Z = sch.linkage(squareform(D, checks=False), method=method)

    # Apply optimal leaf ordering for smoother visual transitions
    Z = sch.optimal_leaf_ordering(Z, squareform(D, checks=False))

    if figsize is None:
        figsize = (0.5 * n + 3.0, 0.5 * n + 3.0)
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 5], width_ratios=[1, 0.045],
                          hspace=0.03, wspace=0.03)
    axd = fig.add_subplot(gs[0, 0])
    axh = fig.add_subplot(gs[1, 0])
    cax = fig.add_subplot(gs[1, 1])

    # Sort branches so smaller groups fall to the left
    dinfo = sch.dendrogram(Z, ax=axd, no_labels=True, color_threshold=0,
                           above_threshold_color="0.4", count_sort="ascending")
    axd.set_xlim(0, 10 * n)
    axd.axis("off")

    # Extract the ordered leaves directly from the dendrogram
    order = dinfo["leaves"]
    cols_ord = [cols[i] for i in order]

    # Apply significance mask where p > 0.05 becomes 0 for display only
    C_masked = np.where(P > 0.05, 0.0, C)
    Cor = C_masked[np.ix_(order, order)]

    im = axh.imshow(Cor, extent=[0, 10 * n, 10 * n, 0], aspect="auto",
                    cmap=cmap, vmin=-1, vmax=1)
    ticks = [10 * p + 5 for p in range(n)]
    axh.set_xticks(ticks)
    axh.set_xticklabels(cols_ord, rotation=90, fontsize=8)
    axh.set_yticks(ticks)
    axh.set_yticklabels(cols_ord, fontsize=8)
    axh.set_xlim(0, 10 * n)
    axh.set_ylim(10 * n, 0)
    fig.colorbar(im, cax=cax, label="Spearman R (p < 0.05)")

    groups = None
    if n_groups:
        g = sch.fcluster(Z, t=n_groups, criterion="maxclust")
        g_ord = g[order]
        start = 0
        for k in range(1, n + 1):
            if k == n or g_ord[k] != g_ord[start]:
                lo, hi = 10 * start, 10 * k
                axh.add_patch(plt.Rectangle((lo, lo), hi - lo, hi - lo,
                                            fill=False, edgecolor="black",
                                            linewidth=1.6))
                start = k
        groups = {cols[i]: int(g[i]) for i in range(n)}

    axd.set_title("Correlation clustermap   "
                  "(order by 1 - |r|, color = signed r, p > 0.05 masked)", fontsize=9)
    return fig, {"order": cols_ord, "linkage": Z, "measures": cols,
                 "groups": groups}
