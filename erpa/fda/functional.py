"""Functional data analysis wrappers for event-locked trial curves.

This module extracts event-locked trial curves, runs functional principal
component analysis, applies elastic registration, and returns registration
outputs with trial metadata. It requires the optional FDA dependencies,
including scikit-fda and the registration dependencies used by
``erpa.fda.registration``.
"""

from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from skfda import FDataGrid
from skfda.preprocessing.dim_reduction import FPCA

from erpa.spatiotemporal.curves import event_locked_curves
from erpa.fda.registration import register_all
from erpa.util import as_meta_frame


# ------------------------------------------------------------------
# Functional PCA
# ------------------------------------------------------------------
def run_fpca(
    grid: ArrayLike,
    Y: ArrayLike,
    n_components: int = 4,
) -> Tuple[FDataGrid, FPCA, np.ndarray, np.ndarray]:
    """Run functional principal component analysis on sampled curves.

    Parameters
    ----------
    grid : array-like
        Grid values for the curve domain.
    Y : array-like
        Curve values with shape ``(n_samples, n_grid_points)``.
    n_components : int, optional
        Number of functional principal components to compute.

    Returns
    -------
    fd : FDataGrid
        Functional data object built from ``Y`` and ``grid``.
    fpca : FPCA
        Fitted functional PCA object.
    scores : np.ndarray
        Functional PCA scores for each sample.
    explained_variance_ratio : np.ndarray
        Fraction of variance explained by each component.
    """
    fd = FDataGrid(data_matrix=Y, grid_points=grid)
    fpca = FPCA(n_components=n_components)
    scores = fpca.fit_transform(fd)
    return fd, fpca, scores, fpca.explained_variance_ratio_


# ------------------------------------------------------------------
# Event-locked FDA and a cross-alignment comparison
# ------------------------------------------------------------------

def event_locked_fda(
    trials: Sequence[Dict[str, Any]],
    event: str = "choice_entry",
    key: str = "lin_vel",
    pre_s: float = 0.4,
    post_s: float = 0.4,
    n_comp: int = 3,
    df: Optional[pd.DataFrame] = None,
    max_iter: int = 8,
) -> Dict[str, Any]:
    """Run event-locked FPCA and elastic registration.

    Curves are extracted in a fixed time window around one behavioral event.
    FPCA is run on the time axis in seconds. Elastic registration is run on a
    normalized grid from 0 to 1 because the registration function returns
    warping functions on that domain.

    The returned lag is the residual timing shift after event locking. Because
    each curve uses a fixed real-time window, the lag is not a direct measure of
    reaction time.

    Parameters
    ----------
    trials : sequence of dict
        Trial dictionaries from ``load_session`` or ``build_trials``. Each trial
        must contain the selected event and data series.
    event : str, optional
        Event used for alignment. Common values are ``"center_entry"``,
        ``"center_exit"``, ``"choice_entry"``, and ``"reward_entry"``.
    key : str, optional
        Trial data field used as the curve values.
    pre_s : float, optional
        Seconds before the alignment event included in each curve.
    post_s : float, optional
        Seconds after the alignment event included in each curve.
    n_comp : int, optional
        Number of amplitude, phase, and joint components returned by
        registration feature extraction.
    df : pandas.DataFrame or None, optional
        Behavioral table used to compute condition correlations. If provided,
        it must contain ``"rt"`` and ``"cue"`` columns indexed by the trial
        indices stored in the curve metadata.
    max_iter : int, optional
        Maximum number of iterations used by elastic registration.

    Returns
    -------
    dict
        Dictionary containing the event name, time grid, raw curves, FPCA
        variance fractions, registered curves, warping functions, template,
        lag, amplitude scores, phase scores, joint scores, metadata, and trial
        indices. If ``df`` is provided, the dictionary also contains ``"corr"``
        with Spearman correlations for lag, reaction time, cue luminance, and
        first amplitude score.
    """
    from scipy.stats import spearmanr
    t, Y, meta = event_locked_curves(trials, event=event, key=key,
                                     pre_s=pre_s, post_s=post_s)
    _, _, _, var = run_fpca(t, Y, n_components=min(4, Y.shape[1] - 1))
    g01 = np.linspace(0, 1, Y.shape[1])
    # One elastic registration. register_all shares a single template across the
    # lag, amplitude, and phase, instead of registering twice.
    from erpa.fda.registration import register_all
    R = register_all(g01, Y, n_comp=n_comp, max_iter=max_iter)
    fn, gam, template, lag = R["fn"], R["gam"], R["template"], R["lag"]
    feats = {"amp": R["amp"], "phase": R["phase"], "joint": R["joint"]}
    idx = as_meta_frame(meta)["idx"].to_numpy()

    out = dict(event=event, t=t, Y=Y, fpca_var=var, registered=fn,
               warping=gam, template=template, lag=lag,
               amp=feats["amp"], phase=feats["phase"], joint=feats["joint"],
               meta=meta, idx=idx)

    if df is not None:
        RT = df["rt"].values[idx]
        cue = df["cue"].values[idx]
        amp1 = feats["amp"][:, 0]
        out["corr"] = dict(
            lag_RT=spearmanr(lag, RT).statistic,
            lag_lum=spearmanr(lag, cue).statistic,
            amp1_lum=spearmanr(amp1, cue).statistic,
            amp1_RT=spearmanr(amp1, RT).statistic,
            pc1_var=float(var[0]),
        )
    return out


def compare_event_locks(
    trials: Sequence[Dict[str, Any]],
    df: pd.DataFrame,
    events: Sequence[str] = ("center_entry", "center_exit", "choice_entry"),
    key: str = "lin_vel",
    pre_s: float = 0.4,
    post_s: float = 0.4,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
    """Compare FPCA and registration summaries across event locks.

    This function runs ``event_locked_fda`` once for each event and summarizes
    the first FPCA variance fraction and condition correlations for each event.
    It also returns the full result dictionary for each event.

    Parameters
    ----------
    trials : sequence of dict
        Trial dictionaries from ``load_session`` or ``build_trials``.
    df : pandas.DataFrame
        Behavioral table passed to ``event_locked_fda``. It must contain
        ``"rt"`` and ``"cue"`` columns.
    events : sequence of str, optional
        Events used for alignment. Each event is passed to
        ``event_locked_fda``.
    key : str, optional
        Trial data field used as the curve values.
    pre_s : float, optional
        Seconds before each alignment event included in each curve.
    post_s : float, optional
        Seconds after each alignment event included in each curve.

    Returns
    -------
    summary : pandas.DataFrame
        One row per event. Columns include the number of curves, first FPCA
        variance fraction, lag correlations, and first-amplitude-score
        correlations.
    results : dict
        Dictionary mapping each event name to the full output of
        ``event_locked_fda``.

    Notes
    -----
    ``"center_exit"`` is the movement-onset alignment used by the velocity-peak
    and time-series analyses. The default comparison also includes
    ``"center_entry"`` and ``"choice_entry"``. All events use the same time
    window width and the same registration settings.
    """
    nice = {"center_entry": "center in", "center_exit": "center out",
            "choice_entry": "choice in", "reward_entry": "reward in"}
    results = {}
    rows = []
    for ev in events:
        r = event_locked_fda(trials, event=ev, key=key,
                             pre_s=pre_s, post_s=post_s, df=df)
        results[ev] = r
        c = r["corr"]
        rows.append(dict(event=nice.get(ev, ev), n=r["Y"].shape[0],
                         pc1_var=round(c["pc1_var"], 3),
                         lag_RT=round(c["lag_RT"], 3),
                         lag_lum=round(c["lag_lum"], 3),
                         amp1_lum=round(c["amp1_lum"], 3),
                         amp1_RT=round(c["amp1_RT"], 3)))
    summary = pd.DataFrame(rows).set_index("event")
    return summary, results
