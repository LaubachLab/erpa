"""Correlation clustermap for ERPA scalar measure tables."""

from typing import Any, Dict, List, Optional, Sequence, Tuple, TypedDict, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
import seaborn as sns
from matplotlib.colors import Colormap, LinearSegmentedColormap
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr


# Diverging Okabe-Ito map: blue for negative correlations, vermillion for
# positive correlations, and white at zero.
OKABE_ITO_DIVERGING: LinearSegmentedColormap = LinearSegmentedColormap.from_list(
    "OkabeIto", ["#0072B2", "#FFFFFF", "#D55E00"]
)

# Columns that describe trials or sessions rather than scalar measures.
LABEL_COLUMNS = {
    "participant_id",
    "name",
    "date",
    "session",
    "sess",
    "session_file",
    "idx",
    "absolute_trial",
    "target",
    "choice",
    "error",
    "hit",
    "cue",
    "cue_level",
    "rt",
    "RT",
    "trial_type",
    "treatment",
}

# When both forms are present, retain the normalized measure by default.
REDUNDANT_PAIRS: Tuple[Tuple[str, str], ...] = (
    ("dev_unchosen_cm", "dev_unchosen_norm"),
    ("path_len_cm", "path_len_norm"),
)


class ClustermapInfo(TypedDict):
    """Metadata returned by :func:`corr_clustermap`."""

    order: List[str]
    linkage: np.ndarray
    measures: List[str]
    groups: Optional[Dict[str, int]]


def _resolve_cmap(cmap: Union[str, Colormap]) -> Colormap:
    """Resolve a colormap name or return an existing colormap.

    Parameters
    ----------
    cmap : str or matplotlib.colors.Colormap
        Colormap name or object. ``"okabe_ito"`` selects the package
        diverging map. ``"vlag"`` selects seaborn's diverging map. Other
        strings are passed to Matplotlib.

    Returns
    -------
    matplotlib.colors.Colormap
        Resolved colormap.
    """
    if isinstance(cmap, Colormap):
        return cmap
    if cmap in {"okabe_ito", "OkabeIto"}:
        return OKABE_ITO_DIVERGING
    if cmap == "vlag":
        return sns.color_palette("vlag", as_cmap=True)
    return plt.get_cmap(cmap)


def corr_clustermap(
    table: pd.DataFrame,
    measures: Optional[Sequence[str]] = None,
    prune: bool = True,
    method: str = "average",
    n_groups: Optional[int] = None,
    cmap: Union[str, Colormap] = "okabe_ito",
    figsize: Optional[Tuple[float, float]] = None,
) -> Tuple[plt.Figure, ClustermapInfo]:
    """Plot a Spearman correlation clustermap of scalar measures.

    Measures are ordered by hierarchical clustering of ``1 - |r|``.
    Correlations with ``p > 0.05`` are displayed as zero.

    Parameters
    ----------
    table : pandas.DataFrame
        Scalar measure table, typically returned by ``build_measure_table``.
    measures : sequence of str or None, optional
        Columns to include. If ``None``, all columns not listed in
        ``LABEL_COLUMNS`` are considered.
    prune : bool, optional
        Remove the centimeter form of a measure when its normalized form is
        also present. This option is ignored only when the pair is incomplete.
    method : str, optional
        Linkage method passed to ``scipy.cluster.hierarchy.linkage``.
    n_groups : int or None, optional
        Number of clusters used to outline diagonal blocks. If ``None``, no
        group boundaries are drawn.
    cmap : str or matplotlib.colors.Colormap, optional
        Heatmap colormap. The default is the Okabe-Ito diverging map.
    figsize : tuple of float or None, optional
        Figure size in inches. If ``None``, the size is based on the number of
        measures.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure containing the dendrogram and heatmap.
    info : ClustermapInfo
        Ordered measure names, linkage matrix, included measures, and optional
        cluster assignments.

    Raises
    ------
    ValueError
        If fewer than three usable measures remain after filtering.

    Notes
    -----
    ``peak_ang_vel`` is converted to absolute values before correlation because
    its sign depends on turn direction.
    """
    data = table.copy()
    if "peak_ang_vel" in data.columns:
        data["peak_ang_vel"] = np.abs(data["peak_ang_vel"])

    if measures is None:
        columns = [column for column in data.columns if column not in LABEL_COLUMNS]
    else:
        columns = list(measures)

    if prune:
        for centimeter, normalized in REDUNDANT_PAIRS:
            if centimeter in columns and normalized in columns:
                columns.remove(centimeter)

    matrix = data[columns].dropna()
    columns = [column for column in columns if matrix[column].std() > 0]
    matrix = matrix[columns]
    n_measures = len(columns)
    if n_measures < 3:
        raise ValueError("At least three usable measures are required.")

    corr, p_values = spearmanr(matrix)
    corr = np.asarray(corr, dtype=float)
    p_values = np.asarray(p_values, dtype=float)

    # ``spearmanr`` can return scalars for two-variable inputs.
    if corr.ndim == 0:
        corr_value = float(corr)
        p_value = float(p_values)
        corr = np.array([[1.0, corr_value], [corr_value, 1.0]])
        p_values = np.array([[0.0, p_value], [p_value, 0.0]])

    np.fill_diagonal(corr, 1.0)

    distance = 1.0 - np.abs(corr)
    np.fill_diagonal(distance, 0.0)
    distance = 0.5 * (distance + distance.T)
    condensed = squareform(distance, checks=False)

    linkage = sch.linkage(condensed, method=method)
    linkage = sch.optimal_leaf_ordering(linkage, condensed)

    if figsize is None:
        figsize = (0.5 * n_measures + 3.0, 0.5 * n_measures + 3.0)

    fig = plt.figure(figsize=figsize)
    grid = fig.add_gridspec(
        2,
        2,
        height_ratios=[1, 5],
        width_ratios=[1, 0.045],
        hspace=0.03,
        wspace=0.03,
    )
    dendrogram_ax = fig.add_subplot(grid[0, 0])
    heatmap_ax = fig.add_subplot(grid[1, 0])
    colorbar_ax = fig.add_subplot(grid[1, 1])

    dendrogram = sch.dendrogram(
        linkage,
        ax=dendrogram_ax,
        no_labels=True,
        color_threshold=0,
        above_threshold_color="0.4",
        count_sort="ascending",
    )
    dendrogram_ax.set_xlim(0, 10 * n_measures)
    dendrogram_ax.axis("off")

    order = [int(index) for index in dendrogram["leaves"]]
    ordered_columns = [columns[index] for index in order]

    displayed_corr = np.where(p_values > 0.05, 0.0, corr)
    ordered_corr = displayed_corr[np.ix_(order, order)]

    image = heatmap_ax.imshow(
        ordered_corr,
        extent=[0, 10 * n_measures, 10 * n_measures, 0],
        aspect="auto",
        cmap=_resolve_cmap(cmap),
        vmin=-1,
        vmax=1,
    )
    ticks = [10 * position + 5 for position in range(n_measures)]
    heatmap_ax.set_xticks(ticks)
    heatmap_ax.set_xticklabels(ordered_columns, rotation=90, fontsize=8)
    heatmap_ax.set_yticks(ticks)
    heatmap_ax.set_yticklabels(ordered_columns, fontsize=8)
    heatmap_ax.set_xlim(0, 10 * n_measures)
    heatmap_ax.set_ylim(10 * n_measures, 0)
    fig.colorbar(image, cax=colorbar_ax, label="Spearman r (p < 0.05)")

    groups: Optional[Dict[str, int]] = None
    if n_groups is not None:
        cluster_ids = sch.fcluster(linkage, t=n_groups, criterion="maxclust")
        ordered_cluster_ids = cluster_ids[order]
        start = 0
        for stop in range(1, n_measures + 1):
            if stop == n_measures or ordered_cluster_ids[stop] != ordered_cluster_ids[start]:
                lower = 10 * start
                upper = 10 * stop
                heatmap_ax.add_patch(
                    plt.Rectangle(
                        (lower, lower),
                        upper - lower,
                        upper - lower,
                        fill=False,
                        edgecolor="black",
                        linewidth=1.6,
                    )
                )
                start = stop
        groups = {
            columns[index]: int(cluster_ids[index])
            for index in range(n_measures)
        }

    dendrogram_ax.set_title(
        "Correlation clustermap "
        "(order by 1 - |r|; color = signed r; p > 0.05 masked)",
        fontsize=9,
    )

    info: ClustermapInfo = {
        "order": ordered_columns,
        "linkage": linkage,
        "measures": columns,
        "groups": groups,
    }
    return fig, info
