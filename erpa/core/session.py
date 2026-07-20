"""Session loading and trial construction.

This module loads SLEAP pose data and behavioral CSV files, computes per-frame
kinematic series, aligns data to task events, and builds trial dictionaries. It
also includes helpers for video downsampling, pose cleaning, event-aligned
velocity extraction, and trajectory resampling.
"""

# =============================================================================
# BEHAVIORAL CSV FORMAT
# =============================================================================
# ERPA loads behavioral data from a CSV file with one row per trial.
# Common source-column names are normalized on load with COLUMN_RENAMES.
#
# Required columns (raw name -> ERPA name):
#   latency    -> rt          Reaction time: center exit to choice entry (s)
#   presented  -> target      Port below the target stimulus (0=L, 1=R)
#   response   -> choice      Port the animal entered (0=L, 1=R)
#   trialtype  -> trial_type  Trial category: 0=single offer, 1=dual offer
#                             Omit this column for detection-only sessions.
#
# Columns passed through without renaming:
#   name        Animal identifier
#   date        Session date (YYYY-MM-DD)
#   session     Session label string
#   sess        Session number within the animal
#   time        Trial start time on the MedPC clock (s)
#   sampling    Time spent at the center port before exiting (s)
#   cue         Number of LEDs illuminated (1, 4, or 16)
#   retrieval   Time from choice entry to reward retrieval (s)
#   iti         Inter-trial interval (s)
#
# Columns computed by load_behavior_csv (not in the raw file):
#   absolute_trial  Original row index before any filtering
#   error           1 if target != choice, 0 if target == choice
#
# Additional columns may be present and are ignored unless retained below.
# =============================================================================

import os
import subprocess
import warnings
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import h5py
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

from erpa.core.config import DEFAULT_CONFIG, RigConfig


Trial = Dict[str, Any]
NodeData = Dict[str, np.ndarray]
SessionSeries = Dict[str, Any]
Ports = Dict[str, np.ndarray]
EpochWindows = Mapping[str, Tuple[int, int]]


class _UnsetType:
    """Sentinel type for arguments that defer to ``RigConfig`` values.
    """

    __slots__ = ()


# ==============================================================================
# CONFIGURATION
# ==============================================================================
#
# These module constants are the package defaults. They are sourced from
# erpa.core.config.DEFAULT_CONFIG, which is the single place they are defined.
# To change them for one session, pass a RigConfig to load_session rather
# than editing here.

# Video parameters
DEFAULT_FRAMERATE = DEFAULT_CONFIG.framerate
REDUCED_RESOLUTION = DEFAULT_CONFIG.reduced_resolution

# Velocity calculation parameters
DEFAULT_SMOOTHING_WINDOW = DEFAULT_CONFIG.smoothing_window
DEFAULT_POLYNOMIAL_ORDER = DEFAULT_CONFIG.poly_order
PIXEL_TO_CM_CONVERSION = 0.099930755788  # used when port scaling is unavailable

# Peak detection parameters
MIN_VELOCITY_THRESHOLD = DEFAULT_CONFIG.min_velocity_threshold
MAX_VELOCITY_SEARCH_WINDOW = DEFAULT_CONFIG.max_velocity_search_window
MIN_VELOCITY_SEARCH_WINDOW = DEFAULT_CONFIG.min_velocity_search_window
MIN_REST_VELOCITY = DEFAULT_CONFIG.min_rest_velocity
REST_VELOCITY_FRACTION = DEFAULT_CONFIG.rest_velocity_fraction

# Outlier detection
POSE_OUTLIER_THRESHOLD = DEFAULT_CONFIG.pose_outlier_threshold
ROBUST_ZSCORE_THRESHOLD = DEFAULT_CONFIG.robust_zscore_threshold

# Behavioral column handling
_UNSET = _UnsetType()

# Normalize common column names used by behavioral exports.
# This code block addresses column name inconsistencies in
# some files in the Laubach Lab.
COLUMN_RENAMES = {
    "sample": "sampling",
    "latency": "rt",
    "RT": "rt",
    "presented": "target",
    "response": "choice",
    "trialtype": "trial_type",
    "itis": "iti",
}


# ==============================================================================
# FUNCTIONS
# ==============================================================================
def _clean_pose_outliers(
    positions: np.ndarray,
    threshold: float = 90
) -> np.ndarray:
    """Replace large pose jumps with interpolated values.

    Frames are flagged when a node position is more than ``threshold`` pixels from
    an adjacent frame. Flagged samples are set to NaN and filled by interpolation.

    Parameters
    ----------
    positions : np.ndarray
        Node position array with shape ``(n_frames, 2, 1)``.
    threshold : float, optional
        Maximum accepted frame-to-frame displacement in pixels.

    Returns
    -------
    numpy.ndarray
        Position array with flagged samples replaced by interpolated values.
    """
    Y = positions.copy()

    def is_far(
        pos1: np.ndarray, pos2: np.ndarray, thresh: float
    ) -> np.ndarray:
        """Return whether paired positions differ by more than a threshold."""
        return np.linalg.norm(np.subtract(pos1, pos2), axis=1) > thresh

    # Iteratively find and remove outliers
    while True:
        # Check distance to previous and next frames
        far_from_prev = is_far(Y[:-1], Y[1:], threshold)[1:]
        far_from_next = is_far(Y[1:], Y[:-1], threshold)[:-1]

        if not (np.any(far_from_prev) or np.any(far_from_next)):
            break

        # Find outlier indices
        outliers_prev = set(np.where(far_from_prev)[0] + 1)
        outliers_next = set(np.where(far_from_next)[0] + 1)
        outliers = sorted(outliers_prev | outliers_next)

        # Handle consecutive outliers to avoid interpolation issues
        for i in outliers:
            if i + 2 in outliers and i + 1 not in outliers:
                outliers.append(i + 1)
        outliers = sorted(set(outliers))

        # Replace outliers with NaN and interpolate
        for i in outliers:
            Y[i] = [[np.nan], [np.nan]]
        Y = _fill_missing_values(Y)

    return Y

def _compute_heading_from_pair(
    left: np.ndarray,
    right: np.ndarray,
    dist: float = 50
) -> np.ndarray:
    """Compute a heading point from paired left and right nodes.

    The point is placed on the line perpendicular to the left-right axis at a fixed
    distance from the midpoint.

    Parameters
    ----------
    left : np.ndarray
        Left node position array with shape ``(n_frames, 2, 1)`` or
        ``(n_frames, 2)``.
    right : np.ndarray
        Right node position array with shape ``(n_frames, 2, 1)`` or
        ``(n_frames, 2)``.
    dist : float, optional
        Distance in pixels from the midpoint to the computed heading point.

    Returns
    -------
    numpy.ndarray
        Computed heading-point coordinates with shape ``(n_frames, 2)``."""
    left = left.reshape(-1, 2)
    right = right.reshape(-1, 2)

    midpoint = (left+right)/2
    direction = right-left
    # Compute normal vectors from list of LR vectors via (-y, x)
    normal = np.array([-1*(direction[:,1].reshape(-1,)), direction[:,0].reshape(-1,)]).T

    # Normalize to unit length by grabbing magnitudes for element-wise division
    mag = (normal[:,0]**2+normal[:,1]**2)**0.5
    magnitudes = mag.reshape(-1, 1) * np.ones((mag.shape[0], 2))

    # Unit normal scaled to preference, does not affect calculations
    unit_normal = (normal / magnitudes) * dist

    # Projecting heading point from the midpoint of L and R keypoints
    heading_point = midpoint + unit_normal
    return heading_point.reshape(-1, 2)

def _fill_missing_values(Y: np.ndarray, kind: str = "linear") -> np.ndarray:
    """Fill missing values by interpolation.

    Values are interpolated independently along each flattened coordinate series.
    Leading and trailing missing values are filled with the nearest valid value.

    Parameters
    ----------
    Y : numpy.ndarray
        Array containing possible NaN values. The first axis is interpreted as time.
    kind : str, optional
        Interpolation method passed to ``scipy.interpolate.interp1d``.

    Returns
    -------
    numpy.ndarray
        Array with the same shape as ``Y`` and missing values filled where possible.
    """
    initial_shape = Y.shape
    Y = Y.reshape((initial_shape[0], -1))

    for i in range(Y.shape[-1]):
        y = Y[:, i]
        valid_idx = np.flatnonzero(~np.isnan(y))

        if len(valid_idx) == 0:
            continue

        # Interpolate interior NaNs
        f = interp1d(valid_idx, y[valid_idx], kind=kind,
                     fill_value=np.nan, bounds_error=False)
        nan_idx = np.flatnonzero(np.isnan(y))
        y[nan_idx] = f(nan_idx)

        # Fill remaining edge NaNs with nearest valid values
        mask = np.isnan(y)
        if np.any(mask):
            y[mask] = np.interp(
                np.flatnonzero(mask),
                np.flatnonzero(~mask),
                y[~mask]
            )
        Y[:, i] = y

    return Y.reshape(initial_shape)

def analyze_epoch_peaks(
    epoch_velocity: np.ndarray,
    window_before: int,
    framerate: int = DEFAULT_FRAMERATE,
    min_velocity: float = MIN_VELOCITY_THRESHOLD
) -> pd.DataFrame:
    """Compute peak measures from event-aligned velocity profiles.

    Finds the maximum velocity in the full window and separate maxima before
    and after the alignment event for each trial.

    Parameters
    ----------
    epoch_velocity : np.ndarray
        Event-aligned velocity array with shape ``(n_trials, window_size)``.
    window_before : int
        Number of frames before the alignment event. This sets the event index in
        each row.
    framerate : int, optional
        Video frame rate used to convert frames to seconds.
    min_velocity : float, optional
        Minimum velocity required for a peak measure to be reported.

    Returns
    -------
    pandas.DataFrame
        Table with columns ``peak_velocity``, ``peak_frame``, ``peak_time``,
        ``pre_event_peak``, and ``post_event_peak``. Values are NaN when no peak
        meets ``min_velocity``."""
    n_trials = epoch_velocity.shape[0]

    results = {
        'peak_velocity': np.full(n_trials, np.nan),
        'peak_frame': np.full(n_trials, np.nan),
        'peak_time': np.full(n_trials, np.nan),
        'pre_event_peak': np.full(n_trials, np.nan),
        'post_event_peak': np.full(n_trials, np.nan),
    }

    for i in range(n_trials):
        vel = epoch_velocity[i]

        if np.all(np.isnan(vel)):
            continue

        # Find overall peak
        valid_mask = ~np.isnan(vel)
        if not np.any(valid_mask):
            continue

        peak_idx = np.nanargmax(vel)
        peak_val = vel[peak_idx]

        if peak_val >= min_velocity:
            results['peak_velocity'][i] = peak_val
            results['peak_frame'][i] = peak_idx
            results['peak_time'][i] = (peak_idx - window_before) / framerate

        # Find pre-event peak (approach velocity)
        pre_event = vel[:window_before]
        if np.any(~np.isnan(pre_event)):
            pre_peak = np.nanmax(pre_event)
            if pre_peak >= min_velocity:
                results['pre_event_peak'][i] = pre_peak

        # Find post-event peak (departure velocity)
        post_event = vel[window_before:]
        if np.any(~np.isnan(post_event)):
            post_peak = np.nanmax(post_event)
            if post_peak >= min_velocity:
                results['post_event_peak'][i] = post_peak

    return pd.DataFrame(results)

def average_node_velocities(
    velocities: Sequence[np.ndarray],
    threshold: float = 10
) -> np.ndarray:
    """Average velocity estimates across tracked nodes.

    Compares node velocities frame by frame and excludes a value when it differs
    from the other estimates by at least ``threshold``.

    Parameters
    ----------
    velocities : sequence of np.ndarray
        Velocity arrays from tracked nodes.
    threshold : float, optional
        Difference threshold used to identify a disagreeing node velocity.

    Returns
    -------
    numpy.ndarray
        Averaged velocity array. Missing values are ignored when possible.
    """
    velocities = [np.atleast_1d(v) for v in velocities]
    max_len = max(len(v) for v in velocities)

    result = []
    for i in range(max_len):
        values = [v[i] if i < len(v) else np.nan for v in velocities]
        values = [v for v in values if not np.isnan(v)]

        n = len(values)
        if n == 0:
            result.append(np.nan)
        elif n == 1:
            result.append(values[0])
        elif n == 2:
            a, b = values
            # Use minimum if they differ too much (outlier rejection)
            result.append(min(a, b) if abs(a - b) >= threshold else (a + b) / 2)
        else:  # n == 3
            # Check for one outlier among three values
            diffs = [[abs(values[i] - values[j])
                     for j in range(n) if i != j]
                    for i in range(n)]
            outlier_counts = [sum(d >= threshold for d in d_list)
                            for d_list in diffs]

            if outlier_counts.count(2) == 1:
                # One node is an outlier; average the other two
                outlier_idx = outlier_counts.index(2)
                valid = [v for i, v in enumerate(values) if i != outlier_idx]
                result.append(sum(valid) / len(valid))
            else:
                result.append(sum(values) / n)

    return np.array(result)

def build_trials(
    df: pd.DataFrame,
    series: SessionSeries,
    frame0_time: float,
    pre_s: float = 0.4,
    post_s: float = 0.8,
    anchor: str = "center_entry",
    t0_anchor: Optional[float] = None,
) -> List[Trial]:
    """Build event-aligned trial dictionaries from session series.

    Each trial window starts before center entry and ends after choice entry. The
    ``anchor`` parameter sets time zero on the trial time axis but does not change
    the window bounds.

    Parameters
    ----------
    df : pandas.DataFrame
        Cleaned behavioral table.
    series : dict
        Output from ``compute_session_series``.
    frame0_time : float
        Video time of the first behavioral trial, in seconds.
    pre_s : float, optional
        Seconds included before center entry.
    post_s : float, optional
        Seconds included after choice entry.
    anchor : str, optional
        Event used as time zero. Valid values are ``"center_entry"``,
        ``"center_exit"``, ``"choice_entry"``, and ``"reward_entry"``.
    t0_anchor : float or None, optional
        Behavioral time used as the alignment origin. If ``None``, the value in
        ``df.attrs["t0_anchor"]`` is used when present; otherwise the first row's
        ``time`` value is used.

    Returns
    -------
    list of dict
        Trial dictionaries containing pose arrays, kinematic arrays, event indices,
        event times, trial window bounds, and behavioral fields.

    Raises
    ------
    KeyError
        If ``anchor`` is not a valid event name.

    Notes
    -----
    Segment-based analyses should use windows that contain both center exit and
    choice entry. Event-centered windows can clip one end of the movement segment.
    """
    valid_anchors = ("center_entry", "center_exit", "choice_entry", "reward_entry")
    if anchor not in valid_anchors:
        raise KeyError(f"anchor '{anchor}' invalid. Choose from {valid_anchors}.")

    fps = series["framerate"]
    centroid = series["centroid"]
    head_angle = series["head_angle"]
    lin_vel = series["lin_vel"]
    ang_vel = series["ang_vel"]
    nodes = series.get("nodes")
    n_frames = centroid.shape[0]
    # Anchor pinned to the original first trial, so validity filtering upstream
    # does not shift the frame alignment. Falls back to the first row only when
    # no anchor was recorded.
    if t0_anchor is None:
        t0_anchor = df.attrs.get("t0_anchor", None)
    t0 = t0_anchor if t0_anchor is not None else df["time"].iloc[0]

    def to_frame(event_time: float) -> int:
        return int(np.floor((event_time - t0 + frame0_time) * fps))

    pre_f = int(round(pre_s * fps))
    post_f = int(round(post_s * fps))

    trials = []
    for i in range(len(df)):
        row = df.iloc[i]
        t_center = row["time"]
        t_exit = t_center + row["sampling"]
        t_choice = t_exit + row["rt"]
        t_reward = t_choice + row["retrieval"]

        events = {
            "center_entry": to_frame(t_center),
            "center_exit": to_frame(t_exit),
            "choice_entry": to_frame(t_choice),
            "reward_entry": to_frame(t_reward),
        }
        anchor_frame = events[anchor]
        # Continuous event times on the video clock, before frame snapping. The
        # MedPC clock resolves near 2 ms, the video to one 40 ms frame. Anchoring
        # the axis to the continuous anchor time removes up to one frame of
        # quantization jitter, which sharpens event-locked registration.
        exact = {
            "center_entry": t_center - t0 + frame0_time,
            "center_exit":  t_exit   - t0 + frame0_time,
            "choice_entry": t_choice - t0 + frame0_time,
            "reward_entry": t_reward - t0 + frame0_time,
        }
        exact_anchor = exact[anchor]
        start = events["center_entry"] - pre_f
        end = events["choice_entry"] + post_f
        start = max(0, start)
        end = min(n_frames, end)
        if end - start < 3:
            continue

        sl = slice(start, end)
        # Time axis anchored to the continuous anchor time, not the snapped frame.
        t_axis = (np.arange(start, end) / fps) - exact_anchor
        quant_error = float(exact_anchor - anchor_frame / fps)
        ev_idx = {k: (v - start) for k, v in events.items()}
        # True event times on the trial axis, sub-frame accurate.
        ev_t = {k: float(v - exact_anchor) for k, v in exact.items()}

        trials_td = {
            "idx": i,
            "absolute_trial": int(row["absolute_trial"]),
            "centroid": centroid[sl].copy(),
            "head_angle": head_angle[sl].copy(),
            "lin_vel": lin_vel[sl].copy(),
            "ang_vel": ang_vel[sl].copy(),
            "nodes": (nodes[sl].copy() if nodes is not None else None),
            "t": t_axis,
            "events": ev_idx,
            "event_times": ev_t,
            "quant_error": quant_error,
            "window": (start, end),
            "framerate": fps,
            # behavioral fields
            "target": int(row["target"]),
            "choice": int(row["choice"]),
            "error": int(row["error"]),
            "cue": float(row["cue"]),
            "rt": float(row["rt"]),
            "sampling": float(row["sampling"]),
            "session": str(row["session"]),
        }
        # Retain trial type when present and non-missing.
        if "trial_type" in row and pd.notna(row["trial_type"]):
            trials_td["trial_type"] = int(row["trial_type"])
        trials.append(trials_td)
    return trials

def calculate_angular_measures(
    posterior_nodes: Sequence[np.ndarray],
    heading_node: Optional[np.ndarray] = None,
    fps: int = 25
) -> Tuple[np.ndarray, np.ndarray]:

    """Compute head angle and angular velocity.

    The heading vector is defined from the posterior-node centroid to
    ``heading_node``. If ``heading_node`` is ``None``, the heading point is computed
    from the line perpendicular to the ordered left-right posterior-node pair.

    Parameters
    ----------
    posterior_nodes : sequence of np.ndarray
        Position arrays used to compute the posterior centroid. Each array must have
        shape ``(n_frames, 2, 1)`` or ``(n_frames, 2)``. When ``heading_node`` is
        ``None``, this must contain the ordered pair ``[left_node, right_node]``.
    heading_node : np.ndarray or None, optional
        Position array used as the head of the heading vector. If ``None``, the
        heading point is computed from ``posterior_nodes``.
    fps : int, optional
        Video frame rate in frames per second.

    Returns
    -------
    head_angle_wrapped : numpy.ndarray
        Head angle in radians, wrapped to the interval ``[-pi, pi]``. Shape is
        ``(n_frames,)``.
    angular_velocity : np.ndarray
        Angular velocity in radians per second. Shape is usually ``(n_frames, 1)``.

    Warns
    -----
    UserWarning
        Warns when ``heading_node`` is ``None`` because the left-right node order
        determines the forward direction.
    """
    def wrap_to_pi(angles: np.ndarray) -> np.ndarray:
        """Wrap an angle series after removing 2-pi discontinuities."""
        angles = angles.copy()
        angles = np.unwrap(angles).reshape(-1,1)
        corrections = (np.round(np.diff(angles, axis=0) / (2 * np.pi)) * (2 * np.pi)).reshape(-1,1)
        angles[1:] += np.cumsum(corrections).T.reshape(-1, 1)
        return angles
    dt = 1/fps
    # Heading vector: posterior landmark to ear midpoint
    posterior_centroid = (sum(posterior_nodes) / len(posterior_nodes)).reshape(-1, 2)
    if heading_node is None:
        warnings.warn(
            "When heading_node is None, posterior_nodes must contain two "
            "symmetric node arrays ordered as (left_node, right_node).",
            stacklevel=2,
        )
        heading_node = _compute_heading_from_pair(posterior_nodes[0], posterior_nodes[1])
    heading = heading_node - posterior_centroid # shape (n_frames, 2)

    # Head angle in radians, relative to positive x-axis
    head_angle = np.arctan2(heading[:, 1], heading[:, 0])
    head_angle_wrapped = (head_angle+ np.pi) % (2*np.pi) - np.pi
    # Unwrap to remove 2-pi discontinuities before differencing
    head_angle_unwrapped = np.unwrap(head_angle)
    head_angle_pi = wrap_to_pi(head_angle_unwrapped)

    # Angular velocity in radians per second
    angular_velocity = np.gradient(head_angle_pi.T, dt, axis=-1)
    angular_velocity = angular_velocity.T
    return head_angle_wrapped, angular_velocity

def calculate_velocity(
    positions: np.ndarray,
    window: int = DEFAULT_SMOOTHING_WINDOW,
    poly_order: int = DEFAULT_POLYNOMIAL_ORDER
) -> np.ndarray:
    """Compute velocity magnitude from position samples.

    Applies a Savitzky-Golay derivative to each spatial coordinate and returns
    the Euclidean norm of the derivative.

    Parameters
    ----------
    positions : np.ndarray
        Position array with shape ``(n_frames, 2, n_tracks)`` or ``(n_frames, 2)``.
        If a track axis is present, only the first track is used.
    window : int, optional
        Savitzky-Golay window length. If an even value is given, one sample is added
        to make it odd.
    poly_order : int, optional
        Polynomial order for the Savitzky-Golay filter.

    Returns
    -------
    numpy.ndarray
        Velocity magnitude with shape ``(n_frames,)`` in position units per frame.

    Notes
    -----
    Callers that need cm/s should multiply by the pixels-to-cm scale and the frame
    rate."""
    # Handle different input shapes
    if positions.ndim == 3:
        pos = positions[:, :, 0]  # Take first track
    else:
        pos = positions

    # Ensure window is odd
    if window % 2 == 0:
        window += 1

    # Compute smoothed derivatives for each dimension
    velocity_components = np.zeros_like(pos)
    for dim in range(pos.shape[-1]):
        try:
            velocity_components[:, dim] = savgol_filter(
                pos[:, dim], window, poly_order, deriv=1
            )
        except ValueError:
            # Fall back to lower polynomial order if window too small
            velocity_components[:, dim] = savgol_filter(
                pos[:, dim], window, min(poly_order, 2), deriv=1
            )

    # Return velocity magnitude
    return np.linalg.norm(velocity_components, axis=1)

def compute_event_frames(
    df: pd.DataFrame,
    event_type: str,
    frame0_time: float,
    framerate: int = DEFAULT_FRAMERATE
) -> np.ndarray:
    """Compute video frame indices for behavioral events.

    Converts behavioral timestamps to video frame indices for one event type.

    Parameters
    ----------
    df : pandas.DataFrame
        Behavioral table containing ``time``, ``sampling``, ``RT``, and
        ``retrieval`` columns.
    event_type : str
        Event to convert. Valid values are ``"center_entry"``, ``"center_exit"``,
        ``"choice_entry"``, and ``"reward_entry"``.
    frame0_time : float
        Video time, in seconds, corresponding to the first trial.
    framerate : int, optional
        Video frame rate in frames per second.

    Returns
    -------
    numpy.ndarray
        Frame indices for the requested event.

    Raises
    ------
    ValueError
        If ``event_type`` is not one of the valid event names.

    Notes
    -----
    Event times are defined from the behavioral columns as follows: center entry is
    ``time``; center exit is ``time + sampling``; choice entry is
    ``time + sampling + RT``; reward entry is ``time + sampling + RT + retrieval``."""
    base_time = df['time'].iloc[0]

    if event_type == 'center_entry':
        event_times = df['time']
    elif event_type == 'center_exit':
        event_times = df['time'] + df['sampling']
    elif event_type == 'choice_entry':
        event_times = df['time'] + df['sampling'] + df['RT']
    elif event_type == 'reward_entry':
        event_times = df['time'] + df['sampling'] + df['RT'] + df['retrieval']
    else:
        raise ValueError(
            f"Unknown event_type: {event_type}. "
            f"Valid options: 'center_entry', 'center_exit', 'choice_entry', 'reward_entry'"
        )

    frames = np.floor((event_times - base_time + frame0_time) * framerate)
    return frames.values

def compute_mean_squared_displacement(
    trials: np.ndarray,
    reference: np.ndarray
) -> np.ndarray:
    """Compute mean squared displacement from a reference trajectory.

    Parameters
    ----------
    trials : np.ndarray
        Trial trajectories with shape ``(n_trials, n_frames, 2)``.
    reference : np.ndarray
        Reference trajectory with shape ``(n_frames, 2)``.

    Returns
    -------
    numpy.ndarray
        Mean squared displacement for each trial, with shape ``(n_trials,)``."""
    diffs = trials - reference[None, :, :]
    sq_dists = np.sum(diffs**2, axis=2)
    return np.mean(sq_dists, axis=1)

def compute_robust_zscore(data: np.ndarray) -> np.ndarray:
    """Compute robust z-scores using the median absolute deviation.

    Parameters
    ----------
    data : np.ndarray
        Values to transform.

    Returns
    -------
    numpy.ndarray
        Robust z-scores computed as ``0.6745 * (data - median) / MAD``. Returns
        zeros when the median absolute deviation is zero."""
    data = np.array(data)
    median = np.median(data)
    mad = np.median(np.abs(data - median))

    if mad == 0:
        return np.zeros_like(data, dtype=float)

    return 0.6745 * (data - median) / mad

def compute_session_series(
    node_data: NodeData,
    framerate: int = DEFAULT_FRAMERATE,
    smoothing_window: int = DEFAULT_SMOOTHING_WINDOW,
    poly_order: int = DEFAULT_POLYNOMIAL_ORDER,
    posterior_nodes: Sequence[str] = ("LeftEar", "RightEar"),
    heading_node: Optional[str] = "MidCann",
    centroid_nodes: Optional[Sequence[str]] = None,
    pixel_to_cm: Optional[float] = None,
) -> SessionSeries:
    """Compute per-frame pose and kinematic series.

    Computes centroid position, head angle, linear velocity, and angular
    velocity for the full session.

    Parameters
    ----------
    node_data : dict of str to numpy.ndarray
        Output from ``load_sleap_data``. Each value has shape ``(n_frames, 2, 1)``.
    framerate : int, optional
        Video frame rate in frames per second.
    smoothing_window : int, optional
        Savitzky-Golay window length for linear velocity.
    poly_order : int, optional
        Polynomial order for Savitzky-Golay velocity smoothing.
    posterior_nodes : sequence of str, optional
        Node names used to compute the posterior centroid. If ``heading_node`` is
        ``None``, this must be the ordered pair ``(left_node, right_node)``.
    heading_node : str or None, optional
        Node name used as the head of the heading vector. Pass ``None`` for data
        with only left and right posterior nodes.
    centroid_nodes : sequence of str or None, optional
        Node names averaged to compute the centroid. If ``None``, all nodes are
        averaged.
    pixel_to_cm : float or None, optional
        Scale factor for converting pixels to centimeters in linear velocity. If
        ``None``, ``PIXEL_TO_CM_CONVERSION`` is used.

    Returns
    -------
    dict
        Dictionary containing ``centroid``, ``head_angle``, ``lin_vel``,
        ``ang_vel``, ``framerate``, ``pixel_to_cm``, ``nodes``, and ``node_names``.

    Raises
    ------
    KeyError
        If ``heading_node`` is not ``None`` and is not present in ``node_data``."""
    names = list(node_data.keys())
    if heading_node is not None and heading_node not in names:
        raise KeyError(
            f"heading_node '{heading_node}' not in {names}. "
            f"For ear-only data pass heading_node=None."
        )
    stack = np.stack([node_data[n][:, :, 0] for n in names], axis=0)  # (nodes, n, 2)

    cen_names = list(centroid_nodes) if centroid_nodes is not None else names
    cen_stack = np.stack([node_data[n][:, :, 0] for n in cen_names], axis=0)
    centroid = cen_stack.mean(axis=0)  # (n, 2)

    # Linear velocity from the centroid, in cm/s.
    try:
        lin = calculate_velocity(centroid, smoothing_window, poly_order)
    except ValueError:
        lin = calculate_velocity(centroid, smoothing_window, 2)
    cm_per_px = PIXEL_TO_CM_CONVERSION if pixel_to_cm is None else pixel_to_cm
    lin_vel = lin * cm_per_px * framerate

    # Angular measures.
    post = [node_data[n][:, :, 0] for n in posterior_nodes]
    head = node_data[heading_node][:, :, 0] if heading_node is not None else None
    head_angle, ang_vel = calculate_angular_measures(post, head, fps=framerate)
    head_angle = np.asarray(head_angle).reshape(-1)
    ang_vel = np.asarray(ang_vel).reshape(-1)
    # Pad angular velocity to n_frames if the routine returns n-1 samples.
    if ang_vel.shape[0] == centroid.shape[0] - 1:
        ang_vel = np.concatenate([ang_vel, ang_vel[-1:]])

    return {
        "centroid": centroid,
        "head_angle": head_angle,
        "lin_vel": lin_vel,
        "ang_vel": ang_vel,
        "framerate": framerate,
        "pixel_to_cm": cm_per_px,
        "nodes": np.transpose(stack, (1, 0, 2)),  # (n, n_nodes, 2)
        "node_names": names,
    }

def downsample_videos(
    input_folder: str,
    output_folder: str,
    output_suffix: str = "_Reduced",
    scale: Tuple[int, int] = REDUCED_RESOLUTION
) -> None:
    """Downsample MP4 videos with FFmpeg.

    Resizes MP4 files from ``input_folder`` and writes the results to
    ``output_folder``. Files that already contain ``output_suffix`` are skipped.

    Parameters
    ----------
    input_folder : str
        Folder containing input MP4 files.
    output_folder : str
        Folder for downsampled MP4 files. The folder is created if it does not
        exist.
    output_suffix : str, optional
        Suffix added to each output file stem.
    scale : tuple of int, optional
        Output resolution as ``(width, height)`` in pixels.

    Notes
    -----
    FFmpeg must be installed and available on the system path. Audio streams are
    removed from the output files."""
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.endswith(".mp4") and output_suffix not in filename:
            input_path = os.path.join(input_folder, filename)
            base_name = os.path.splitext(filename)[0]
            output_filename = f"{base_name}{output_suffix}.mp4"
            output_path = os.path.join(output_folder, output_filename)

            cmd = [
                "ffmpeg", "-y", "-i", input_path,
                "-vf", f"scale={scale[0]}:{scale[1]}",
                "-an",  # remove audio
                output_path
            ]

            print(f"Processing: {filename} → {output_filename}")
            subprocess.run(cmd, check=True)

def estimate_frame0_time(
    df: pd.DataFrame,
    series: SessionSeries,
    search: Optional[Tuple[float, float, float]] = None,
    coarse_step: float = 0.5,
    fine_step: float = 0.02,
    move_window_s: float = 0.3,
) -> float:
    """Estimate the video time of the first behavioral trial.

    Scores candidate video offsets using the velocity contrast between the
    center-port hold period and the movement period after center exit. The estimate
    is intended for alignment checks; determine ``frame0_time`` from the video for
    final analyses whenever possible.

    Parameters
    ----------
    df : pandas.DataFrame
        Behavioral table containing ``time``, ``sampling``, and ``RT`` columns.
    series : dict
        Output from ``compute_session_series``.
    search : tuple of float or None, optional
        ``(start, stop, step)`` values for the offset scan in seconds. If ``None``,
        a coarse scan is followed by a fine scan.
    coarse_step : float, optional
        Step size, in seconds, for the coarse scan when ``search`` is ``None``.
    fine_step : float, optional
        Step size, in seconds, for the fine scan when ``search`` is ``None``.
    move_window_s : float, optional
        Length of the movement-onset window after center exit, in seconds.

    Returns
    -------
    float
        Estimated ``frame0_time`` in seconds.

    Notes
    -----
    The estimate covers the video offset only and does not include
    ``trial_offset``. Verify the result against the source video.
    """
    fps = series["framerate"]
    sv = np.asarray(series["lin_vel"], float)
    n = sv.shape[0]
    dur = n / fps
    csum = np.concatenate([[0.0], np.nancumsum(sv)])
    move_frames = max(1, int(round(move_window_s * fps)))

    t0 = df.attrs.get("t0_anchor", None)
    if t0 is None:
        t0 = df["time"].iloc[0]
    t_entry = (df["time"] - t0).to_numpy(float)
    t_exit = (df["time"] + df["sampling"] - t0).to_numpy(float)
    t_choice = (df["time"] + df["sampling"] + df["rt"] - t0).to_numpy(float)

    def window_mean(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        # mean of sv over [a, b) per trial, O(1) via the cumulative sum
        good = b > a
        out = np.full(a.shape, np.nan)
        out[good] = (csum[b[good]] - csum[a[good]]) / (b[good] - a[good])
        return out

    def score(f0: float) -> float:
        fe = np.floor((t_entry + f0) * fps).astype(int)
        fx = np.floor((t_exit + f0) * fps).astype(int)
        fc = np.floor((t_choice + f0) * fps).astype(int)
        fm = np.minimum(fc, fx + move_frames)   # movement-onset window
        ok = (fe > 1) & (fc < n - 1) & (fx > fe) & (fm > fx)
        if ok.sum() < 5:
            return -np.inf
        hold = window_mean(fe[ok], fx[ok])
        move = window_mean(fx[ok], fm[ok])
        return np.nanmean(move) - np.nanmean(hold)

    def scan(grid: np.ndarray) -> float:
        best_f0, best = grid[0], -np.inf
        for f0 in grid:
            s = score(f0)
            if s > best:
                best, best_f0 = s, f0
        return best_f0

    if search is not None:
        return float(scan(np.arange(*search)))

    last = np.nanmax(t_choice)
    hi = max(coarse_step, dur - last - 1.0)
    coarse = scan(np.arange(0.0, hi, coarse_step))
    lo = max(0.0, coarse - coarse_step)
    fine = scan(np.arange(lo, coarse + coarse_step, fine_step))
    return float(fine)

def extract_epoch_velocities(
    sleap_file: str,
    behavioral_csv: str,
    frame0_time: float,
    trial_offset: int = 0,
    smoothing_window: int = DEFAULT_SMOOTHING_WINDOW,
    poly_order: int = DEFAULT_POLYNOMIAL_ORDER,
    framerate: int = DEFAULT_FRAMERATE,
    epochs: Optional[EpochWindows] = None
) -> Tuple[Dict[str, np.ndarray], pd.DataFrame]:
    """Extract velocity profiles for multiple task epochs.

    Loads pose and behavioral data, computes full-session velocity from the
    first tracked node, and extracts event-aligned windows.

    Parameters
    ----------
    sleap_file : str
        Path to a SLEAP ``.analysis.h5`` file.
    behavioral_csv : str
        Path to a behavioral CSV file containing ``time``, ``sampling``, ``RT``, and
        ``retrieval`` columns.
    frame0_time : float
        Video time, in seconds, corresponding to the first trial.
    trial_offset : int, optional
        Number of leading behavioral rows to skip.
    smoothing_window : int, optional
        Savitzky-Golay window length for velocity calculation.
    poly_order : int, optional
        Polynomial order for the Savitzky-Golay filter.
    framerate : int, optional
        Video frame rate in frames per second.
    epochs : dict or None, optional
        Mapping from epoch name to ``(window_before, window_after)`` in frames. If
        ``None``, default windows are used for center entry, center exit, choice
        entry, and reward entry.

    Returns
    -------
    epoch_velocities : dict of str to numpy.ndarray
        Mapping from epoch name to velocity array with shape
        ``(n_trials, window_before + window_after)``.
    df : pandas.DataFrame
        Behavioral table after ``trial_offset`` is applied.

    Notes
    -----
    Velocity is scaled with ``PIXEL_TO_CM_CONVERSION``. This helper does not derive a
    per-session scale from detected ports."""
    # Default epoch windows (in frames at 25fps)
    # Tuned for typical movement durations in the task
    if epochs is None:
        epochs = {
            'center_entry': (38, 25),   # 1.5s before (approach), 1s after (sampling)
            'center_exit': (13, 38),    # 0.5 s before, 1.5 s after
            'choice_entry': (25, 25),   # 1s before (choice approach), 1s after (exit to reward)
            'reward_entry': (38, 13),   # 1.5s before (travel from choice), 0.5s after (at reward)
        }

    # Load pose data and compute session-wide velocity
    node_data = load_sleap_data(sleap_file)
    primary_node = list(node_data.values())[0]  # Use first node (typically cannula)

    try:
        session_velocity = calculate_velocity(primary_node[:, :, 0], smoothing_window, poly_order)
    except ValueError:
        session_velocity = calculate_velocity(primary_node[:, :, 0], smoothing_window, 2)

    # Convert to cm/s
    session_velocity = session_velocity * PIXEL_TO_CM_CONVERSION * framerate

    # Load behavioral data
    df = pd.read_csv(behavioral_csv, index_col=0)
    df = df.iloc[trial_offset:].reset_index(drop=True)

    # Extract velocity for each epoch
    epoch_velocities = {}
    for epoch_name, (window_before, window_after) in epochs.items():
        event_frames = compute_event_frames(df, epoch_name, frame0_time, framerate)
        epoch_velocities[epoch_name] = get_event_aligned_velocity(
            session_velocity, event_frames, window_before, window_after
        )

    return epoch_velocities, df

def find_peak_velocity(
    velocity: np.ndarray,
    search_start: int = 0,
    min_velocity: float = MIN_VELOCITY_THRESHOLD,
    max_search_frames: int = MAX_VELOCITY_SEARCH_WINDOW
) -> Tuple[Optional[float], Optional[int]]:
    """Find the first local velocity maximum after a start frame.

    Searches forward from ``search_start`` for the first local maximum above
    ``min_velocity`` within ``max_search_frames``.

    Parameters
    ----------
    velocity : np.ndarray
        Velocity time series for one trial.
    search_start : int, optional
        Frame index where the search begins. Passing the center-exit frame excludes
        peaks before the choice movement.
    min_velocity : float, optional
        Minimum velocity required for a local maximum to count as a peak.
    max_search_frames : int, optional
        Maximum number of frames searched after ``search_start``.

    Returns
    -------
    peak_velocity : float or None
        Peak velocity value, or ``None`` if no peak is found.
    peak_index : int or None
        Frame index of the peak, or ``None`` if no peak is found."""
    if velocity.size == 0:
        return None, None

    search_start = max(1, search_start)
    search_end = min(len(velocity) - 1, search_start + max_search_frames)

    for i in range(search_start, search_end):
        if (velocity[i - 1] < velocity[i] > velocity[i + 1] and
                velocity[i] > min_velocity):
            return float(velocity[i]), i

    return None, None

def find_velocity_minimum(
    velocity: np.ndarray,
    peak_index: Optional[int] = None,
    min_velocity: float = MIN_REST_VELOCITY,
    max_search_frames: int = MIN_VELOCITY_SEARCH_WINDOW,
    rest_fraction: float = REST_VELOCITY_FRACTION,
) -> int:
    """Find a local velocity minimum near the start of a trial.

    Searches the first ``max_search_frames`` for local minima below the selected
    floor. When ``peak_index`` is provided, the last qualifying minimum before the
    peak is preferred.

    Parameters
    ----------
    velocity : np.ndarray
        Velocity time series for one trial.
    peak_index : int or None, optional
        Peak frame index used to select a preceding minimum.
    min_velocity : float, optional
        Maximum velocity for a local minimum to qualify.
    max_search_frames : int, optional
        Maximum number of frames searched from the start of the trial.
    rest_fraction : float, optional
        Fraction of the peak velocity used as the rest floor when ``peak_index``
        is given. The absolute ``min_velocity`` floor is used when no peak is
        given.

    Returns
    -------
    int
        Frame index of the selected local minimum. Returns 0 when no qualifying
        minimum is found."""
    search_end = min(len(velocity) - 2, max_search_frames)

    # Rest floor relative to the trial's own peak when a peak is given, so the
    # test is scale-free across sessions and boxes rather than tied to the
    # absolute cm scale. The absolute min_velocity floor is the fallback when no
    # peak is supplied. The 0.2 default approximates the prior 6 cm/s floor at a
    # typical peak speed.
    if peak_index is not None and 0 <= peak_index < len(velocity):
        floor = rest_fraction * float(velocity[peak_index])
    else:
        floor = min_velocity

    # Find local minima below the floor
    minima = []
    for i in range(1, search_end):
        if (velocity[i-1] > velocity[i] < velocity[i+1] and
            velocity[i] < floor):
            minima.append(i)

    if not minima:
        return 0

    # Return the minimum that precedes the peak
    if peak_index is not None:
        valid_minima = [m for m in minima if m < peak_index]
        if valid_minima:
            return max(valid_minima)

    return minima[0] if minima else 0

def get_event_aligned_velocity(
    session_velocity: np.ndarray,
    event_frames: np.ndarray,
    window_before: int = 25,
    window_after: int = 25
) -> np.ndarray:
    """Extract velocity windows aligned to event frames.

    Each output row contains a fixed-length segment from ``session_velocity`` around
    one event. Values outside the session bounds are filled with NaN.

    Parameters
    ----------
    session_velocity : np.ndarray
        Full-session velocity array.
    event_frames : np.ndarray
        Video frame indices for alignment events.
    window_before : int, optional
        Number of frames before the event frame.
    window_after : int, optional
        Number of frames from the event frame through the end of the extracted
        window.

    Returns
    -------
    numpy.ndarray
        Array with shape ``(n_events, window_before + window_after)``. The event
        frame is placed at index ``window_before`` when it lies inside the session."""
    n_events = len(event_frames)
    window_size = window_before + window_after
    aligned = np.full((n_events, window_size), np.nan)

    for i, frame in enumerate(event_frames):
        frame = int(frame)
        start = frame - window_before
        end = frame + window_after

        # Calculate valid range within session bounds
        valid_start = max(0, start)
        valid_end = min(len(session_velocity), end)

        if valid_start >= valid_end:
            continue  # Event is completely outside session

        # Extract the valid segment
        segment = session_velocity[valid_start:valid_end]

        # Calculate offset for placing segment in output array
        output_start = valid_start - start  # How many frames we missed at the beginning
        aligned[i, output_start:output_start + len(segment)] = segment

    return aligned

def load_behavior_csv(
    behavioral_csv: str,
    trial_offset: int = 0,
    valid_pct: float = 95.0,
) -> pd.DataFrame:
    """Load and clean a behavioral CSV file.

    Normalizes task column names, computes the error indicator, and filters
    invalid trials.

    Parameters
    ----------
    behavioral_csv : str
        Path to the behavioral CSV file.
    trial_offset : int, optional
        Number of leading trials to drop before validity filtering.
    valid_pct : float, optional
        Upper percentile cutoff for ``sampling`` and ``rt``. Trials above either
        cutoff are removed.

    Returns
    -------
    pandas.DataFrame
        Cleaned behavioral table with one row per kept trial. Columns are:
        ``name``, ``date``, ``session``, ``sess``, ``trial_type``, ``sampling``,
        ``rt``, ``cue``, ``target``, ``choice``, ``retrieval``, ``iti``,
        ``absolute_trial``, and ``error``.

    Notes
    -----
    ``error`` is computed as ``target != choice``: 1 for an error, 0 for a
    correct trial. This definition applies to both single-offer and dual-offer
    trials. Alignment uses timestamps from retained trials and the original first
    trial time stored in ``df.attrs["t0_anchor"]``.
    """
    df = pd.read_csv(behavioral_csv, index_col=0)
    df = df.rename(columns=COLUMN_RENAMES)
    # Drop duplicate column names that arise when a file contains both the
    # original name (e.g. latency) and the target name (e.g. rt). Keep the
    # first occurrence, which is the renamed canonical column.
    df = df.loc[:, ~df.columns.duplicated(keep="first")]
    # Retain the source-row index before slicing or filtering. This value supports
    # traceability and sequential analyses but does not affect alignment.
    df["absolute_trial"] = np.asarray(df.index)
    df = df.iloc[trial_offset:].reset_index(drop=True)

    # Store the first trial time before validity filtering. ``frame0_time`` is
    # calibrated to this trial, so filtering must not shift the alignment anchor.
    t0_anchor = float(df["time"].iloc[0]) if len(df) else None

    # Count before any validity drop, so the dropped-trial summary is complete.
    n0 = len(df)

    # Drop trials with missing target, choice, or trial_type. An aborted trial
    # can leave these blank, and the int casts in build_trials would fail.
    df = df[df["target"].notna() & df["choice"].notna()]
    if "trial_type" in df.columns:
        df = df[df["trial_type"].notna()]
    df = df.reset_index(drop=True)

    # error is 1 when target and choice differ, 0 when they match.
    # This definition is correct for both single-offer and dual-offer trials.
    df["error"] = (df["target"].astype(int) != df["choice"].astype(int)).astype(int)

    # Validity filters.
    df = df[df["sampling"] >= 0]                        # negative = MedPC artifact
    if len(df):
        samp_hi = np.nanpercentile(df["sampling"], valid_pct)
        rt_hi = np.nanpercentile(df["rt"], valid_pct)
        df = df[(df["sampling"] <= samp_hi) & (df["rt"] <= rt_hi)]
    df = df.reset_index(drop=True)
    df.attrs["t0_anchor"] = t0_anchor
    n_drop = n0 - len(df)
    if n_drop:
        print(
            f"load_behavior_csv dropped {n_drop} of {n0} trials with a missing "
            f"outcome, negative sampling, or sampling/rt above the "
            f"{valid_pct:.0f}th percentile."
        )

    # Return only the columns needed for ERPA analysis.
    keep = ["name", "date", "session", "sess", "time", "trial_type", "sampling",
            "rt", "cue", "target", "choice", "retrieval", "iti",
            "absolute_trial", "error"]
    keep = [c for c in keep if c in df.columns]
    return df[keep]


def load_session(
    sleap_file: str,
    behavioral_csv: str,
    frame0_time: Optional[float] = None,
    trial_offset: int = 0,
    config: Optional[RigConfig] = None,
    pre_s: Union[float, _UnsetType] = _UNSET,
    post_s: Union[float, _UnsetType] = _UNSET,
    posterior_nodes: Union[Sequence[str], _UnsetType] = _UNSET,
    heading_node: Union[str, None, _UnsetType] = _UNSET,
    centroid_nodes: Union[Sequence[str], None, _UnsetType] = _UNSET,
) -> Tuple[List[Trial], pd.DataFrame, SessionSeries, Ports, float]:
    """Load pose and behavior data for one ERPA session.

    Loads SLEAP and behavioral data, computes session-level pose and kinematic
    series, builds trial dictionaries, estimates port locations, and recalculates
    velocity with a port-derived scale when both choice ports are available.

    Parameters
    ----------
    sleap_file : str
        Path to a SLEAP ``.analysis.h5`` file.
    behavioral_csv : str
        Path to the behavioral CSV file.
    frame0_time : float or None, optional
        Video time of the first behavioral trial, in seconds. This value is
        required; passing ``None`` raises ``ValueError``.
    trial_offset : int, optional
        Number of leading behavioral trials to drop before alignment.
    config : RigConfig or None, optional
        Rig and pose-model settings. If ``None``, ``DEFAULT_CONFIG`` is used.
    pre_s : float, optional
        Seconds included before center entry. The config value is used when this
        argument is not supplied.
    post_s : float, optional
        Seconds included after choice entry. The config value is used when this
        argument is not supplied.
    posterior_nodes : sequence of str, optional
        Node names used to compute the posterior centroid. The config value is used
        when this argument is not supplied.
    heading_node : str or None, optional
        Node name used as the head of the heading vector. Pass ``None`` for ear-only
        data. The config value is used when this argument is not supplied.
    centroid_nodes : sequence of str or None, optional
        Node names averaged to compute the centroid. The config value is used when
        this argument is not supplied.

    Returns
    -------
    trials : list of dict
        Trial dictionaries from ``build_trials``.
    df : pandas.DataFrame
        Cleaned behavioral table from ``load_behavior_csv``.
    series : dict
        Session-level pose and kinematic series from ``compute_session_series``.
    ports : dict of str to numpy.ndarray
        Estimated port coordinates from ``locate_ports``.
    frame0_time : float
        Video time of the first behavioral trial, in seconds.

    Raises
    ------
    ValueError
        If ``frame0_time`` is ``None``."""
    cfg = config if config is not None else DEFAULT_CONFIG
    pre_s = cfg.pre_s if pre_s is _UNSET else pre_s
    post_s = cfg.post_s if post_s is _UNSET else post_s
    posterior_nodes = cfg.posterior_nodes if posterior_nodes is _UNSET else posterior_nodes
    heading_node = cfg.heading_node if heading_node is _UNSET else heading_node
    centroid_nodes = cfg.centroid_nodes if centroid_nodes is _UNSET else centroid_nodes

    if frame0_time is None:
        raise ValueError(
            "frame0_time must be provided. Determine frame0_time from the video "
            "before loading the session. estimate_frame0_time is available as an "
            "experimental helper, but it is not used automatically."
        )
    frame0_time = float(frame0_time)

    node_data = load_sleap_data(
        sleap_file, outlier_threshold=cfg.pose_outlier_threshold)
    df = load_behavior_csv(behavioral_csv, trial_offset=trial_offset,
                           valid_pct=cfg.valid_pct)

    series_kw = dict(framerate=cfg.framerate,
                     smoothing_window=cfg.smoothing_window,
                     poly_order=cfg.poly_order,
                     posterior_nodes=posterior_nodes,
                     heading_node=heading_node,
                     centroid_nodes=centroid_nodes)

    # Build a provisional series to estimate port locations. Centroid positions
    # are in pixels, so the provisional velocity scale does not affect alignment.
    series = compute_session_series(node_data, **series_kw)
    trials = build_trials(df, series, frame0_time, pre_s=pre_s, post_s=post_s)
    ports = locate_ports(trials)

    # When both choice ports are available, derive a session-specific spatial
    # scale and rebuild the velocity-dependent trial fields.
    from erpa.spatiotemporal.spatial import cm_per_pixel
    if {"choice_L", "choice_R"} <= set(ports):
        pix_cm = cm_per_pixel(ports, port_spacing_cm=cfg.port_spacing_cm)
        series = compute_session_series(node_data, pixel_to_cm=pix_cm, **series_kw)
        trials = build_trials(df, series, frame0_time, pre_s=pre_s, post_s=post_s)
    else:
        warnings.warn("Both choice ports were not located, so velocity stays on "
                      "the PIXEL_TO_CM_CONVERSION scale.")

    return trials, df, series, ports, frame0_time

def load_sleap_data(
    sleap_file: str,
    outlier_threshold: float = POSE_OUTLIER_THRESHOLD,
) -> NodeData:
    """Load pose data from a SLEAP analysis HDF5 file.

    Reads tracked node locations, fills missing values, and replaces large
    frame-to-frame jumps with interpolated values.

    Parameters
    ----------
    sleap_file : str
        Path to a SLEAP ``.analysis.h5`` file.
    outlier_threshold : float, optional
        Pixel displacement threshold used to flag a frame as a tracking outlier.

    Returns
    -------
    dict of str to numpy.ndarray
        Mapping from node name to position array. Each array has shape
        ``(n_frames, 2, 1)`` and stores x and y coordinates.

    Notes
    -----
    Missing values are filled by linear interpolation. Leading and trailing missing
    values are filled with the nearest valid value."""
    with h5py.File(sleap_file, "r") as f:
        locations = f["tracks"][:].T  # Transpose to (frames, nodes, xy, tracks)
        node_names = [n.decode() for n in f["node_names"][:]]

    # Interpolate missing values
    locations = _fill_missing_values(locations)

    # Build dictionary of node positions
    node_data = {}
    for i, name in enumerate(node_names):
        node_data[name] = locations[:, i, :, :]

    # Clean outliers from each node's trajectory
    for node in node_data:
        node_data[node] = _clean_pose_outliers(
            node_data[node],
            threshold=outlier_threshold
        )

    return node_data

def locate_ports(trials: Sequence[Trial]) -> Ports:
    """Estimate port locations from trial event positions.

    The center port is estimated from centroid positions at center entry. The left
    and right choice ports are estimated from centroid positions at choice entry,
    split by target side.

    Parameters
    ----------
    trials : list of dict
        Trial dictionaries from ``build_trials``.

    Returns
    -------
    dict of str to numpy.ndarray
        Mapping from port label to x-y coordinates. Possible keys are ``center``,
        ``choice_L``, and ``choice_R``. Missing ports are omitted."""
    bins = {"center": [], "choice_L": [], "choice_R": []}
    for t in trials:
        cen = t["centroid"]
        ce = t["events"]["center_entry"]
        ch = t["events"]["choice_entry"]
        if 0 <= ce < len(cen):
            bins["center"].append(cen[ce])
        if 0 <= ch < len(cen):
            key = "choice_R" if t["choice"] == 1 else "choice_L"
            bins[key].append(cen[ch])
    out = {}
    for k, vals in bins.items():
        if vals:
            out[k] = np.nanmedian(np.array(vals), axis=0)
    return out

def ports_array(ports: Mapping[str, np.ndarray]) -> np.ndarray:
    """Stack port coordinates into a fixed-order array.

    Parameters
    ----------
    ports : dict of str to numpy.ndarray
        Mapping from port label to x-y coordinates.

    Returns
    -------
    numpy.ndarray
        Port coordinates with shape ``(n_ports, 2)`` ordered as ``center``,
        ``choice_L``, and ``choice_R`` when present."""
    order = [k for k in ("center", "choice_L", "choice_R") if k in ports]
    return np.array([ports[k] for k in order])

def prepare_figure_trials(
    trials: Sequence[Trial],
    velocity_key: str = "lin_vel",
) -> List[Trial]:
    """Add fields used by trajectory and velocity plotting functions.

    Returns shallow copies of trial dictionaries with plotting fields for the
    trajectory, velocity, outcome, and velocity extrema.

    Parameters
    ----------
    trials : list of dict
        Trial dictionaries from ``build_trials`` or ``load_session``.
    velocity_key : str, optional
        Trial key copied into the ``velocity`` field. Default is ``"lin_vel"``.

    Returns
    -------
    list of dict
        Shallow-copied trial dictionaries with ``trajectory``, ``velocity``,
        ``error``, ``localmax``, ``lmax_idx``, and ``localmin`` fields added."""
    out = []
    for t in trials:
        vel = np.asarray(t[velocity_key], float)
        center_exit = t["events"].get("center_exit", 0)
        peak_vel, peak_idx = find_peak_velocity(vel, search_start=center_exit)
        min_idx = (find_velocity_minimum(vel, peak_idx)
                   if peak_idx is not None else 0)
        d = dict(t)
        d["trajectory"] = t["centroid"]
        d["velocity"] = vel
        d["error"] = int(t["error"])
        d["localmax"] = peak_vel
        d["lmax_idx"] = peak_idx
        d["localmin"] = min_idx
        out.append(d)
    return out

def resample_trajectory(
    trajectory: np.ndarray,
    original_length: int,
    target_length: int = 100
) -> np.ndarray:
    """Resample a trajectory to a fixed number of samples.

    Parameters
    ----------
    trajectory : np.ndarray
        Position array with shape ``(n_frames, 2)``.
    original_length : int
        Number of valid samples from ``trajectory`` to use.
    target_length : int, optional
        Number of samples in the output trajectory.

    Returns
    -------
    numpy.ndarray
        Resampled trajectory with shape ``(target_length, 2)``."""
    x = np.linspace(0, 1, original_length)
    x_new = np.linspace(0, 1, target_length)

    interp_func = interp1d(x, trajectory[:original_length], axis=0)
    return interp_func(x_new)

def resample_velocity(
    velocity: np.ndarray,
    original_length: int,
    target_length: int = 100
) -> np.ndarray:
    """Resample a velocity profile to a fixed number of samples.

    Parameters
    ----------
    velocity : np.ndarray
        Velocity array.
    original_length : int
        Number of valid samples from ``velocity`` to use.
    target_length : int, optional
        Number of samples in the output velocity profile.

    Returns
    -------
    numpy.ndarray
        Resampled velocity array with shape ``(target_length,)`` for one-dimensional
        input."""
    x = np.linspace(0, 1, original_length)
    x_new = np.linspace(0, 1, target_length)

    interp_func = interp1d(x, velocity[:original_length], axis=0)
    return interp_func(x_new)

def reshape_trajectories(
    trials: Sequence[Trial],
    output_path: Optional[str] = None
) -> np.ndarray:
    """Pad trial trajectories to a common length.

    Converts variable-length trial trajectories to a three-dimensional array
    with NaN padding.

    Parameters
    ----------
    trials : list of dict
        Trial dictionaries containing a ``trajectory`` array.
    output_path : str or None, optional
        Path for saving the output as a ``.npy`` file. If ``None``, no file is
        written.

    Returns
    -------
    numpy.ndarray
        Array with shape ``(n_trials, max_length, 2)``."""
    max_len = max(len(t['trajectory']) for t in trials)
    n_trials = len(trials)

    output = np.full((n_trials, max_len, 2), np.nan)

    for i, trial in enumerate(trials):
        traj = trial['trajectory']
        if traj.ndim == 3:
            traj = traj[:, :, 0]  # Remove track dimension
        output[i, :len(traj), :] = traj

    if output_path:
        np.save(output_path, output)

    return output

def synchronize_timescales(
    behavioral_times: np.ndarray,
    frame0_time: float,
    framerate: int = DEFAULT_FRAMERATE,
    use_decay_model: bool = False,
    decay_coefficient: float = 29.407498,
    dropped_frames_start: float = 0,
    dropped_frames_interval: int = 6
) -> Tuple[np.ndarray, float]:
    """Convert behavioral times to video frame indices.

    Converts behavioral timestamps to video frames using either the nominal
    frame rate or a linear drift coefficient.

    Parameters
    ----------
    behavioral_times : np.ndarray
        Behavioral event times in seconds.
    frame0_time : float
        Video time, in seconds, corresponding to the first behavioral timestamp.
    framerate : int, optional
        Nominal video frame rate in frames per second.
    use_decay_model : bool, optional
        If ``True``, use ``decay_coefficient`` instead of ``framerate`` for the time
        conversion.
    decay_coefficient : float, optional
        Effective frame-rate coefficient used when ``use_decay_model`` is ``True``.
    dropped_frames_start : float, optional
        Fraction of the session after which dropped-frame compensation is applied.
        A value of 0 disables this compensation.
    dropped_frames_interval : int, optional
        Interval used in the dropped-frame compensation term.

    Returns
    -------
    frame_indices : numpy.ndarray
        Video frame indices for ``behavioral_times``.
    effective_framerate : float
        Frame rate used for the time conversion."""
    times = behavioral_times

    # Calculate dropped frame compensation
    if dropped_frames_start > 0:
        drop_threshold = float(times[-1]) * dropped_frames_start
        extra_drops = (np.fix(times / drop_threshold) *
                      np.round((times - times[0]) / (dropped_frames_interval * 100)))
    else:
        extra_drops = 0

    if use_decay_model:
        # Linear model for videos with drift
        trial_starts = np.fix(-frame0_time + times * decay_coefficient) - 1 + extra_drops
        effective_framerate = np.round(decay_coefficient)
    else:
        # Standard alignment
        trial_starts = np.floor((times - times[0] + frame0_time) * framerate) + extra_drops
        effective_framerate = framerate

    return trial_starts, effective_framerate
