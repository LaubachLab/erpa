"""Plotting functions for functional-data summaries.

This module contains plotting functions for functional principal component
analysis, elastic shape analysis, and event-locked registration summaries.
"""

from os import PathLike
from typing import Any, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike


OKABE_ITO = {
    "orange": "#E69F00",
    "sky_blue": "#56B4E9",
    "bluish_green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "reddish_purple": "#CC79A7",
    "black": "#000000",
}

# High-contrast sequence for categorical values. Yellow is omitted because it is
# hard to see on a white plotting background when used for thin lines or points.
OKABE_CYCLE = [
    OKABE_ITO["blue"],
    OKABE_ITO["orange"],
    OKABE_ITO["bluish_green"],
    OKABE_ITO["vermillion"],
    OKABE_ITO["reddish_purple"],
    OKABE_ITO["sky_blue"],
    OKABE_ITO["black"],
]


def _okabe_colors(n: int) -> list[str]:
    """Return ``n`` colors from the Okabe-Ito categorical palette."""
    if n <= 0:
        return []
    return [OKABE_CYCLE[i % len(OKABE_CYCLE)] for i in range(n)]


def _despine(ax: "matplotlib.axes.Axes") -> None:
    """Remove top and right spines from an axes object."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def group_strip(
    ax: "matplotlib.axes.Axes",
    values: ArrayLike,
    group: np.ndarray,
    groups: Sequence[Any],
    labels: Sequence[str],
    ylabel: str,
) -> None:
    """Plot jittered values by group with a group mean.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes object to draw on.
    values : array-like
        Numeric values to plot.
    group : array-like
        Group labels for each value.
    groups : sequence
        Group labels to plot, in display order.
    labels : sequence of str
        Tick labels for the plotted groups.
    ylabel : str
        Label for the y-axis.

    Returns
    -------
    None
        The function modifies ``ax`` in place.
    """
    values = np.asarray(values, dtype=float)
    group = np.asarray(group)

    rng = np.random.default_rng(0)
    for i, g in enumerate(groups):
        v = values[group == g]
        v = v[np.isfinite(v)]
        x = i + rng.uniform(-0.12, 0.12, size=v.size)
        ax.scatter(x, v, s=10, alpha=0.35, color=OKABE_ITO["black"])
        if v.size:
            ax.hlines(
                np.mean(v),
                i - 0.25,
                i + 0.25,
                color=OKABE_ITO["blue"],
                lw=2.5,
            )
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.axhline(0, color=OKABE_ITO["black"], ls=":", lw=0.8, alpha=0.55)
    _despine(ax)


def plot_fpca_modes(
    grid: ArrayLike,
    mean: ArrayLike,
    modes: Sequence[Tuple[ArrayLike, ArrayLike]],
    scores: np.ndarray,
    var_ratio: ArrayLike,
    error: np.ndarray,
    cue: np.ndarray,
    ylabel: str,
) -> "matplotlib.figure.Figure":
    """Plot functional PCA modes and related score summaries.

    The first row shows the mean curve and the positive and negative mode
    curves for up to the first three components. The second row shows the
    variance ratio, PC1 scores by outcome, and mean scores by cue level.

    Parameters
    ----------
    grid : array-like
        Grid values for the curves.
    mean : array-like
        Mean curve on ``grid``.
    modes : sequence of tuple of array-like
        Positive and negative mode curves for each component.
    scores : array-like
        Functional PCA scores with shape ``(n_trials, n_components)``.
    var_ratio : array-like
        Fraction of variance explained by each component.
    error : array-like
        Outcome indicator for each trial. A value of 1 marks an error.
    cue : array-like
        Cue value or cue level for each trial.
    ylabel : str
        Label for the curve y-axis.

    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the mode and score summary plots.
    """
    import matplotlib.pyplot as plt

    grid = np.asarray(grid)
    mean = np.asarray(mean)
    scores = np.asarray(scores)
    var_ratio = np.asarray(var_ratio)
    error = np.asarray(error)
    cue = np.asarray(cue)

    k = len(modes)
    fig, ax = plt.subplots(2, 3, figsize=(13, 8), dpi=100)

    for j in range(min(3, k)):
        up, dn = modes[j]
        ax[0, j].plot(
            grid,
            mean,
            color=OKABE_ITO["black"],
            lw=2,
            label="mean",
        )
        ax[0, j].plot(
            grid,
            up,
            color=OKABE_ITO["blue"],
            lw=1.8,
            label="+",
        )
        ax[0, j].plot(
            grid,
            dn,
            color=OKABE_ITO["vermillion"],
            lw=1.8,
            label="-",
        )
        ax[0, j].set_title(f"PC{j + 1} mode ({var_ratio[j] * 100:.0f}%)")
        ax[0, j].set_xlabel("normalized distance")
        ax[0, j].set_ylabel(ylabel)
        _despine(ax[0, j])
        if j == 0:
            ax[0, j].legend(fontsize=8, frameon=False)

    ax[1, 0].bar(
        np.arange(1, k + 1),
        var_ratio[:k],
        color=OKABE_ITO["blue"],
    )
    ax[1, 0].set_title("variance by mode")
    ax[1, 0].set_xlabel("PC")
    ax[1, 0].set_ylabel("fraction")
    _despine(ax[1, 0])

    group_strip(
        ax[1, 1],
        scores[:, 0],
        (error == 1).astype(int),
        [0, 1],
        ["correct", "error"],
        "PC1 score",
    )
    ax[1, 1].set_title("PC1 by outcome")

    cues = sorted(np.unique(cue))
    for j, col in zip([0, 1], [OKABE_ITO["blue"], OKABE_ITO["vermillion"]]):
        if j < scores.shape[1]:
            mu = [np.nanmean(scores[cue == c, j]) for c in cues]
            ax[1, 2].plot(
                [int(c) for c in cues],
                mu,
                "o-",
                color=col,
                label=f"PC{j + 1}",
            )
    ax[1, 2].set_title("score vs cue")
    ax[1, 2].set_xlabel("cue level")
    ax[1, 2].set_ylabel("mean score")
    ax[1, 2].legend(fontsize=8, frameon=False)
    _despine(ax[1, 2])

    # Hide unused top-row axes if fewer than three modes are available.
    for j in range(k, 3):
        ax[0, j].set_visible(False)

    fig.tight_layout()
    return fig


def plot_shape_modes(
    mean_shape: np.ndarray,
    modes: np.ndarray,
    scores: np.ndarray,
    var_ratio: ArrayLike,
    error: np.ndarray,
    cue: np.ndarray,
    unit: str,
) -> "matplotlib.figure.Figure":
    """Plot elastic shape modes and related score summaries.

    The figure shows the Karcher mean shape, the first two principal shape
    modes, PC1 scores by outcome, and PC1-PC2 score plots colored by outcome
    and cue.

    Parameters
    ----------
    mean_shape : array-like
        Mean shape with shape ``(n_points, 2)``.
    modes : array-like
        Shape mode array with shape ``(2, n_points, n_components, n_steps)``.
    scores : array-like
        Shape PCA scores with shape ``(n_trials, n_components)``.
    var_ratio : array-like
        Fraction of variance explained by each shape component.
    error : array-like
        Outcome indicator for each trial. A value of 1 marks an error.
    cue : array-like
        Cue value or cue level for each trial.
    unit : str
        Unit label for the spatial axes.

    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the shape mode and score summary plots.
    """
    import matplotlib.pyplot as plt

    mean_shape = np.asarray(mean_shape)
    modes = np.asarray(modes)
    scores = np.asarray(scores)
    var_ratio = np.asarray(var_ratio)
    error = np.asarray(error)
    cue = np.asarray(cue)

    fig, ax = plt.subplots(2, 3, figsize=(13, 8), dpi=100)

    ax[0, 0].plot(
        mean_shape[:, 0],
        mean_shape[:, 1],
        color=OKABE_ITO["black"],
        lw=2.5,
    )
    ax[0, 0].scatter(
        [mean_shape[0, 0]],
        [mean_shape[0, 1]],
        color=OKABE_ITO["bluish_green"],
        zorder=5,
        label="start",
    )
    ax[0, 0].scatter(
        [mean_shape[-1, 0]],
        [mean_shape[-1, 1]],
        color=OKABE_ITO["vermillion"],
        zorder=5,
        label="end",
    )
    ax[0, 0].set_title("Karcher mean shape (unit length)")
    ax[0, 0].set_aspect("equal", adjustable="datalim")
    ax[0, 0].locator_params(axis="both", nbins=4)
    ax[0, 0].legend(fontsize=8, frameon=False)
    ax[0, 0].set_xlabel(f"perpendicular ({unit})")
    ax[0, 0].set_ylabel(f"toward chosen ({unit})")
    _despine(ax[0, 0])

    nsteps = modes.shape[3]
    step_colors = _okabe_colors(nsteps)
    for jj, axj in zip([0, 1], [ax[0, 1], ax[0, 2]]):
        for st in range(nsteps):
            axj.plot(
                modes[0, :, jj, st],
                modes[1, :, jj, st],
                color=step_colors[st],
                lw=1,
                alpha=0.85,
            )
        axj.set_title(f"shape mode PC{jj + 1} ({var_ratio[jj] * 100:.0f}%)")
        axj.set_aspect("equal", adjustable="datalim")
        axj.locator_params(axis="both", nbins=4)
        axj.set_xlabel(f"perpendicular ({unit})")
        _despine(axj)

    group_strip(
        ax[1, 0],
        scores[:, 0],
        (error == 1).astype(int),
        [0, 1],
        ["correct", "error"],
        "shape PC1",
    )
    ax[1, 0].set_title("shape PC1 by outcome")

    def _rlim(a: "matplotlib.axes.Axes") -> None:
        xlo, xhi = np.nanpercentile(scores[:, 0], [1, 99])
        ylo, yhi = np.nanpercentile(scores[:, 1], [1, 99])
        mx = 0.1 * (xhi - xlo + 1e-9)
        my = 0.1 * (yhi - ylo + 1e-9)
        a.set_xlim(xlo - mx, xhi + mx)
        a.set_ylim(ylo - my, yhi + my)

    outcome_colors = {0: OKABE_ITO["blue"], 1: OKABE_ITO["vermillion"]}
    outcome_labels = {0: "correct", 1: "error"}
    outcome_group = (error == 1).astype(int)
    for g in [0, 1]:
        mask = outcome_group == g
        ax[1, 1].scatter(
            scores[mask, 0],
            scores[mask, 1],
            color=outcome_colors[g],
            s=14,
            alpha=0.7,
            label=outcome_labels[g],
        )
    ax[1, 1].set_title("scores by outcome (1-99 pct)")
    ax[1, 1].set_xlabel("PC1")
    ax[1, 1].set_ylabel("PC2")
    ax[1, 1].legend(fontsize=8, frameon=False)
    _rlim(ax[1, 1])
    _despine(ax[1, 1])

    cues = sorted(np.unique(cue))
    cue_colors = _okabe_colors(len(cues))
    for c, col in zip(cues, cue_colors):
        mask = cue == c
        try:
            label = f"cue {int(c)}"
        except Exception:
            label = f"cue {c}"
        ax[1, 2].scatter(
            scores[mask, 0],
            scores[mask, 1],
            color=col,
            s=14,
            alpha=0.7,
            label=label,
        )
    ax[1, 2].set_title("scores by cue (1-99 pct)")
    ax[1, 2].set_xlabel("PC1")
    ax[1, 2].set_ylabel("PC2")
    ax[1, 2].legend(fontsize=8, frameon=False, markerscale=1.2)
    _rlim(ax[1, 2])
    _despine(ax[1, 2])

    fig.tight_layout()
    return fig


def plot_event_lock_comparison(
    results: Mapping[str, Mapping[str, Any]],
    summary: Optional["pandas.DataFrame"] = None,
    save_path: Optional[Union[str, PathLike[str]]] = None,
) -> "matplotlib.figure.Figure":
    """Plot event-locked curves and registered curves for several alignments.

    The top row shows velocity traces and their mean on the real-time axis, with
    time zero at the alignment event. The bottom row shows registered amplitude
    curves and their mean on a normalized time axis.

    Parameters
    ----------
    results : dict
        Maps event names to result dictionaries from
        ``erpa.fda.functional.event_locked_fda``.
    summary : pandas.DataFrame or None, optional
        Summary table from ``erpa.fda.functional.compare_event_locks``.
        This argument is accepted for caller compatibility and is not used for
        drawing.
    save_path : str or None, optional
        Path for saving the figure. If ``None``, the figure is not saved.

    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the event-lock comparison.
    """
    import matplotlib.pyplot as plt

    nice = {
        "center_entry": "center in",
        "center_exit": "center out",
        "choice_entry": "choice in",
        "reward_entry": "reward in",
    }
    events = list(results)
    n = len(events)
    fig, ax = plt.subplots(2, n, figsize=(4.2 * n, 6.4), dpi=130, sharey="row")

    if n == 1:
        ax = ax.reshape(2, 1)

    for j, ev in enumerate(events):
        r = results[ev]
        t, Y, fn = r["t"], r["Y"], r["registered"]

        ax[0, j].plot(
            t,
            Y.T,
            color=OKABE_ITO["black"],
            alpha=0.14,
            lw=0.5,
        )
        ax[0, j].plot(
            t,
            np.nanmean(Y, axis=0),
            color=OKABE_ITO["blue"],
            lw=2.5,
        )
        ax[0, j].axvline(
            0,
            color=OKABE_ITO["orange"],
            lw=1.2,
            ls="--",
        )
        ax[0, j].set_title(f"locked to {nice.get(ev, ev)}")
        ax[0, j].set_xlabel("time from event (s)")
        _despine(ax[0, j])

        g01 = np.linspace(0, 1, Y.shape[1])
        ax[1, j].plot(
            g01,
            fn.T,
            color=OKABE_ITO["black"],
            alpha=0.14,
            lw=0.5,
        )
        ax[1, j].plot(
            g01,
            np.nanmean(fn, axis=0),
            color=OKABE_ITO["bluish_green"],
            lw=2.5,
        )
        ax[1, j].set_title("registered amplitude")
        ax[1, j].set_xlabel("registered time")
        _despine(ax[1, j])

    ax[0, 0].set_ylabel("cm/s")
    ax[1, 0].set_ylabel("cm/s")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
    return fig
