"""
Curve extraction from ERPA trial dictionaries.

This module extracts movement-segment curves and event-locked curves from
trial dictionaries. The functions resample selected per-trial signals to common
grids for scalar summaries and optional functional data analyses.
"""

import warnings
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from erpa.core.session import PIXEL_TO_CM_CONVERSION
from erpa.util import as_meta_frame


Trial = Dict[str, Any]


def movement_curves(
    trials: Sequence[Trial],
    key: str = "lin_vel",
    n_points: int = 50,
    pad: int = 2,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Extract movement-segment curves and resample them to a common grid.

    The movement segment runs from center exit to choice entry. Each segment is
    linearly resampled to ``n_points`` samples on a normalized grid from 0 to 1.

    Parameters
    ----------
    trials : list of dict
        Trial dictionaries from ``build_trials`` or ``load_session``. Each trial
        must contain ``events`` and the signal named by ``key``.
    key : str, optional
        Trial field to extract and resample.
    n_points : int, optional
        Number of samples in each resampled curve.
    pad : int, optional
        Number of frames added before center exit and after choice entry.

    Returns
    -------
    grid : numpy.ndarray
        Normalized grid with shape ``(n_points,)``.
    Y : numpy.ndarray
        Resampled curves with shape ``(n_kept, n_points)``.
    meta : pandas.DataFrame
        Trial metadata for the kept curves.

    Notes
    -----
    The input trials must contain the full center-entry-to-choice-entry span.
    Trials are skipped when center exit or choice entry is clipped by the trial
    window. A nonzero skipped-trial count indicates that the trials should be
    rebuilt on the full span.
    """
    grid = np.linspace(0, 1, n_points)
    rows, meta = [], []
    n_clipped = 0
    for t in trials:
        ce = t["events"]["center_exit"]
        ch = t["events"]["choice_entry"]
        # The segment must sit inside the window. Center exit clipped at the
        # start, or choice entry pushed outside the window, both mean the window
        # was centered on one event instead of spanning the trial. A benign tail
        # truncation at the recording boundary keeps choice entry inside the
        # window and is not flagged.
        if ce < pad or ch >= len(t[key]):
            n_clipped += 1
            continue
        a = max(0, ce - pad)
        b = min(len(t[key]), ch + pad)
        seg = t[key][a:b]
        if seg.size < 5 or not np.all(np.isfinite(seg)):
            continue
        x = np.linspace(0, 1, seg.size)
        rows.append(np.interp(grid, x, seg))
        meta_row = {k: t[k] for k in ("idx", "target", "choice",
                                      "cue", "rt", "error") if k in t}
        if "trial_type" in t:
            meta_row["trial_type"] = t["trial_type"]
        meta.append(meta_row)
    if n_clipped:
        warnings.warn(
            f"movement_curves skipped {n_clipped} trials with a clipped segment. "
            f"The movement window must span center entry to choice entry. "
            f"Rebuild trials on the full span, not a window centered on one event."
        )
    return grid, np.array(rows), as_meta_frame(meta)


def node_speed(
    nodes: ArrayLike,
    framerate: float,
    pix_to_cm: float,
) -> np.ndarray:
    """
    Compute per-node speed from position samples.

    Speed is frame-to-frame displacement scaled by the frame rate and the
    pixels-to-centimeters factor.

    Parameters
    ----------
    nodes : numpy.ndarray
        Position array with shape ``(n_frames, n_nodes, 2)``.
    framerate : int or float
        Video frame rate in frames per second.
    pix_to_cm : float
        Conversion factor from pixels to centimeters.

    Returns
    -------
    numpy.ndarray
        Per-node speed in cm/s with shape ``(n_frames, n_nodes)``.
    """
    d = np.diff(nodes, axis=0)  # (n-1, n_nodes, 2)
    sp = np.sqrt((d ** 2).sum(axis=2)) * pix_to_cm * framerate
    sp = np.vstack([sp, sp[-1:]])  # pad to n_frames
    return sp  # (n_frames, n_nodes)


def keypoint_speed_curves(
    trials: Sequence[Trial],
    node_names: Sequence[str],
    n_points: int = 50,
    pad: int = 2,
) -> Tuple[np.ndarray, np.ndarray, list[str], pd.DataFrame]:
    """
    Extract per-keypoint speed curves over the movement segment.

    The movement segment runs from center exit to choice entry. For each kept
    trial, node positions are converted to speed and resampled to a common
    normalized grid.

    Parameters
    ----------
    trials : list of dict
        Trial dictionaries from ``build_trials`` or ``load_session``. Each trial
        must contain ``nodes``, ``events``, and ``framerate``.
    node_names : sequence of str
        Names of the pose nodes, in the order used by the trial ``nodes`` array.
    n_points : int, optional
        Number of samples in each resampled curve.
    pad : int, optional
        Number of frames added before center exit and after choice entry.

    Returns
    -------
    grid : numpy.ndarray
        Normalized grid with shape ``(n_points,)``.
    Y : numpy.ndarray
        Resampled speed curves with shape ``(n_kept, n_points, n_nodes)``.
    names : list of str
        Pose node names.
    meta : pandas.DataFrame
        Trial metadata for the kept curves.

    Notes
    -----
    This function uses ``PIXEL_TO_CM_CONVERSION`` for the pixel-to-centimeter
    scale. Trials are skipped when the movement segment is clipped by the trial
    window.
    """
    pix = PIXEL_TO_CM_CONVERSION
    grid = np.linspace(0, 1, n_points)
    rows, meta = [], []
    n_clipped = 0
    for t in trials:
        if t.get("nodes") is None:
            continue
        m = t["nodes"].shape[0]
        ce = t["events"]["center_exit"]
        ch = t["events"]["choice_entry"]
        if ce < pad or ch >= m:
            n_clipped += 1
            continue
        a = max(0, ce - pad)
        b = min(m, ch + pad)
        blk = t["nodes"][a:b]
        if blk.shape[0] < 6 or not np.all(np.isfinite(blk)):
            continue
        sp = node_speed(blk, t["framerate"], pix)  # (m, n_nodes)
        x = np.linspace(0, 1, sp.shape[0])
        cols = [np.interp(grid, x, sp[:, j]) for j in range(sp.shape[1])]
        rows.append(np.stack(cols, axis=1))  # (n_points, n_nodes)
        meta_row = {k: t[k] for k in ("idx", "target", "cue",
                                      "error") if k in t}
        if "trial_type" in t:
            meta_row["trial_type"] = t["trial_type"]
        meta.append(meta_row)
    if n_clipped:
        warnings.warn(
            f"keypoint_speed_curves skipped {n_clipped} trials with a clipped "
            f"segment. Rebuild trials on the full center-entry-to-choice-entry "
            f"span, not a window centered on one event."
        )
    return grid, np.array(rows), list(node_names), as_meta_frame(meta)


def event_locked_curves(
    trials: Sequence[Trial],
    event: str = "choice_entry",
    key: str = "lin_vel",
    pre_s: float = 0.4,
    post_s: float = 0.4,
    fps: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Extract fixed-duration curves around one behavioral event.

    Each trial contributes a real-time window with zero at the selected event.
    The duration is not normalized, so the output retains timing in seconds.

    Parameters
    ----------
    trials : list of dict
        Trial dictionaries from ``build_trials`` or ``load_session``.
    event : str, optional
        Event name in each trial's ``events`` dictionary.
    key : str, optional
        Trial field to extract.
    pre_s : float, optional
        Seconds to include before the event.
    post_s : float, optional
        Seconds to include after the event.
    fps : int, float, or None, optional
        Frame rate in frames per second. If ``None``, the frame rate from the
        first trial is used.

    Returns
    -------
    t_axis : numpy.ndarray
        Time axis in seconds with zero at the event.
    Y : numpy.ndarray
        Event-locked curves with shape ``(n_kept, n_points)``.
    meta : pandas.DataFrame
        Trial metadata for the kept curves.

    Raises
    ------
    KeyError
        If ``event`` is not present in a trial's ``events`` dictionary.

    Notes
    -----
    Trials are skipped when the requested window extends past the trial bounds
    or contains non-finite values. For elastic registration, pass a normalized
    grid to the registration functions. Use ``t_axis`` for plotting and FPCA on
    the real-time axis.
    """
    fps = fps or trials[0]["framerate"]
    pre_f = int(round(pre_s * fps))
    post_f = int(round(post_s * fps))
    t_axis = np.arange(-pre_f, post_f) / fps
    rows, meta, n_skip = [], [], 0
    for t in trials:
        if event not in t["events"]:
            raise KeyError(f"event '{event}' not in {list(t['events'])}.")
        e = t["events"][event]
        a, b = e - pre_f, e + post_f
        if a < 0 or b > len(t[key]):
            n_skip += 1
            continue
        seg = t[key][a:b]
        if not np.all(np.isfinite(seg)):
            n_skip += 1
            continue
        rows.append(seg)
        meta_row = {k: t[k] for k in ("idx", "target", "choice",
                                      "cue", "rt", "error") if k in t}
        if "trial_type" in t:
            meta_row["trial_type"] = t["trial_type"]
        meta.append(meta_row)
    if n_skip:
        warnings.warn(
            f"event_locked_curves skipped {n_skip} trials whose {event} window "
            f"ran past the trial bounds. Rebuild span trials with larger pre_s "
            f"or post_s in load_session to make room."
        )
    return t_axis, np.array(rows), as_meta_frame(meta)
