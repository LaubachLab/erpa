"""Trajectory and velocity plotting functions.

This module contains plotting functions for top-down trajectories, heading
direction, event-aligned velocity, and velocity time series. Orientation transforms
change displayed coordinates only; analytical coordinates in trial dictionaries
are not changed.
"""

from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from numpy.typing import ArrayLike

from erpa.core.session import DEFAULT_FRAMERATE

SIDE_LABELS: Dict[int, str] = {0: "L", 1: "R"}


Trial = Dict[str, Any]
Ports = Optional[Union[np.ndarray, Dict[str, ArrayLike]]]
OrientationName = Literal["raw", "paper"]
KinematicKind = Literal["lin_vel", "ang_vel"]


class ViewTransform:
    """Display transform for arena coordinates.

    The transform applies an integer number of 90 degree rotations about the
    arena center, followed by optional horizontal and vertical flips. The same
    linear transform is applied to points and direction vectors so transformed
    heading directions remain aligned with transformed trajectories.

    Parameters
    ----------
    n_rot : int, optional
        Number of counterclockwise 90 degree rotations. Values are reduced
        modulo 4.
    flip_u : bool, optional
        If True, flip the horizontal display axis.
    flip_v : bool, optional
        If True, flip the vertical display axis.
    """

    def __init__(
        self,
        n_rot: int = 1,
        flip_u: bool = False,
        flip_v: bool = False,
    ) -> None:
        self.n_rot = n_rot % 4
        self.flip_u = flip_u
        self.flip_v = flip_v

    def _matrix(self) -> np.ndarray:
        """Return the 2D linear transform matrix.

        Returns
        -------
        np.ndarray
            Matrix with shape ``(2, 2)``.
        """
        theta = 0.5 * np.pi * self.n_rot
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s], [s, c]])
        F = np.diag([-1.0 if self.flip_u else 1.0, -1.0 if self.flip_v else 1.0])
        return F @ R

    def apply(self, xy: ArrayLike) -> np.ndarray:
        """Apply the transform to points or vectors.

        Parameters
        ----------
        xy : array-like
            Array whose last dimension has length 2.

        Returns
        -------
        np.ndarray
            Transformed array with the same shape as ``xy``.
        """
        xy = np.asarray(xy, dtype=float)
        M = self._matrix()
        return xy @ M.T


# Identity transform: recorded arena coordinates, the "raw" orientation.
RAW_VIEW = ViewTransform(n_rot=0)

# Paper orientation: a 90 degree counterclockwise display rotation that places
# the response ports across the top of the arena, with choice_L on the left.
PORTS_TOP = ViewTransform(n_rot=3, flip_v=True)

# Display orientations selectable by name in the trajectory plotters.
ORIENTATIONS: Dict[OrientationName, ViewTransform] = {
    "raw": RAW_VIEW,
    "paper": PORTS_TOP,
}


def _orientation_transform(
    orientation: OrientationName = "paper",
    transform: Optional[ViewTransform] = None,
) -> ViewTransform:
    """Resolve the display transform for trajectory plots.

    Parameters
    ----------
    orientation : {'raw', 'paper'}, optional
        Named display orientation. ``'raw'`` uses recorded arena coordinates.
        ``'paper'`` places the response ports across the top of the plot.
    transform : ViewTransform or None, optional
        Custom display transform. If not ``None``, this value is returned and
        ``orientation`` is ignored.

    Returns
    -------
    ViewTransform
        Display transform used for plotting.

    Raises
    ------
    ValueError
        If ``orientation`` is not a known orientation name.
    """
    if transform is not None:
        return transform
    try:
        return ORIENTATIONS[orientation]
    except KeyError:
        raise ValueError(
            f"orientation must be one of {sorted(ORIENTATIONS)}, got {orientation!r}.")


def _select_trials(
    trials: Sequence[Trial],
    trial_indices: Optional[Sequence[int]],
) -> List[Trial]:
    """Select trials by behavioral trial index.

    Parameters
    ----------
    trials : list of dict
        Trial dictionaries.
    trial_indices : sequence of int or None
        Behavioral ``idx`` values to return. If ``None``, all trials are
        returned.

    Returns
    -------
    list of dict
        Selected trials in the requested order. Requested indices not present
        in ``trials`` are skipped.
    """
    if trial_indices is None:
        return list(trials)
    by_idx = {t["idx"]: t for t in trials}
    return [by_idx[i] for i in trial_indices if i in by_idx]

OKABE_ITO: Dict[str, str] = {
    "orange": "#E69F00",
    "sky_blue": "#56B4E9",
    "bluish_green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "reddish_purple": "#CC79A7",
    "black": "#000000",
}

OKABE_CYCLE: List[str] = [
    OKABE_ITO["blue"],
    OKABE_ITO["orange"],
    OKABE_ITO["bluish_green"],
    OKABE_ITO["reddish_purple"],
    OKABE_ITO["sky_blue"],
    OKABE_ITO["vermillion"],
    OKABE_ITO["yellow"],
    OKABE_ITO["black"],
]


EVENT_COLORS: Dict[str, str] = {
    "center_entry": OKABE_ITO["orange"],
    "center_exit": OKABE_ITO["reddish_purple"],
    "choice_entry": OKABE_ITO["bluish_green"],
}


def _trial_event_index(trial: Trial, name: str) -> Optional[int]:
    """Return an event index, or None if the event is missing or invalid.

    Parameters
    ----------
    trial : dict
        Trial dictionary.
    name : str
        Event name in ``trial['events']``.

    Returns
    -------
    int or None
        Event frame index, if available.
    """
    events = trial.get("events", {})
    if name not in events:
        return None
    try:
        return int(events[name])
    except (TypeError, ValueError):
        return None


def _ports_to_display_points(
    ports: Ports,
    transform: ViewTransform,
) -> Tuple[np.ndarray, List[str]]:
    """Return transformed port points and labels.

    Parameters
    ----------
    ports : dict, np.ndarray, or None
        Port coordinates in raw arena coordinates.
    transform : ViewTransform
        Display transform applied before plotting.

    Returns
    -------
    np.ndarray
        Transformed port coordinates with shape ``(n_ports, 2)``.
    list of str
        Port labels.
    """
    if ports is None:
        return np.empty((0, 2), dtype=float), []

    if isinstance(ports, dict):
        labels = []
        pts = []
        for label, xy in ports.items():
            arr = np.asarray(xy, dtype=float)
            if arr.shape == (2,):
                labels.append(str(label))
                pts.append(arr)
        if not pts:
            return np.empty((0, 2), dtype=float), []
        return transform.apply(np.vstack(pts)), labels

    arr = np.asarray(ports, dtype=float)
    if arr.ndim == 1 and arr.size == 2:
        arr = arr.reshape(1, 2)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(
            "ports must be a dict of (x, y) values or an array with "
            "shape (n_ports, 2).")

    labels = [f"port {i}" for i in range(arr.shape[0])]
    return transform.apply(arr), labels


def _align_ports(ports: Ports) -> Ports:
    """Align response ports for display.

    The three response ports are projected onto a shared vertical line at the
    mean of their x coordinates. The center port is placed at the midpoint of
    the two side ports along that line. A reward port, if present, is aligned
    to the same shared x. The aligned coordinates are used only for plotting;
    analytical functions must use the original port coordinates.

    Parameters
    ----------
    ports : dict, np.ndarray, or None
        Port coordinates in raw arena coordinates. A dictionary is expected,
        keyed by ``'center'``, ``'choice_L'``, ``'choice_R'``, and optionally
        ``'reward'``. Non-dictionary inputs are returned unchanged, since
        alignment needs named ports.

    Returns
    -------
    dict, np.ndarray, or None
        Aligned copy of ``ports``. Non-dictionary inputs are passed through.
    """
    if not isinstance(ports, dict):
        return ports

    required = ("center", "choice_L", "choice_R")
    if not all(key in ports for key in required):
        return ports

    left = np.asarray(ports["choice_L"], dtype=float)
    right = np.asarray(ports["choice_R"], dtype=float)
    shared_x = float(np.mean([
        np.asarray(ports[key], dtype=float)[0] for key in required
    ]))
    midpoint_y = float(0.5 * (left[1] + right[1]))

    aligned = dict(ports)
    aligned["choice_L"] = np.array([shared_x, left[1]])
    aligned["choice_R"] = np.array([shared_x, right[1]])
    aligned["center"] = np.array([shared_x, midpoint_y])
    if "reward" in ports:
        reward = np.asarray(ports["reward"], dtype=float)
        aligned["reward"] = np.array([shared_x, reward[1]])
    return aligned


def _offset_ports_display(
    port_disp: np.ndarray,
    snout_offset_px: float,
) -> np.ndarray:
    """Shift port markers away from trajectory endpoints for display.

    The tracked centroid stops short of each port by roughly the distance from
    the head marker to the snout, so port estimates read from pose data fall
    inside the trajectory cloud. Markers are shifted by a fixed amount along the
    positive vertical display axis, which points away from the trajectories when
    the response ports are drawn across the top of the plot.
    A fixed vertical shift is exact only when the ports sit across the top of
    the display, which is the ``paper`` orientation. Do not use the offset when
    plotting single trials using raw orientations.

    Parameters
    ----------
    port_disp : np.ndarray
        Transformed port coordinates with shape ``(n_ports, 2)``.
    snout_offset_px : float
        Vertical shift in display pixels. The plotting functions use a
        configurable default that approximates the head-to-snout distance.

    Returns
    -------
    np.ndarray
        Shifted copy of ``port_disp``. Empty input is returned unchanged.
    """
    if port_disp.size == 0:
        return port_disp
    shifted = port_disp.copy()
    shifted[:, 1] = shifted[:, 1] + float(snout_offset_px)
    return shifted


def _display_ports(
    ports: Ports,
    transform: ViewTransform,
    align: bool,
    snout_offset_px: float,
) -> Tuple[np.ndarray, List[str]]:
    """Return display-ready port points, optionally aligned and offset.

    Alignment is applied in raw coordinates, followed by the display
    transform and then the vertical marker offset. Both corrections are
    display-only and are applied only when requested.

    Parameters
    ----------
    ports : dict, np.ndarray, or None
        Port coordinates in raw arena coordinates.
    transform : ViewTransform
        Display transform applied before plotting.
    align : bool
        If True, force the response ports collinear and equally spaced.
    snout_offset_px : float
        Vertical display shift applied to port markers. Zero disables the
        offset.

    Returns
    -------
    np.ndarray
        Display-ready port coordinates with shape ``(n_ports, 2)``.
    list of str
        Port labels.
    """
    prepared = _align_ports(ports) if align else ports
    port_disp, labels = _ports_to_display_points(prepared, transform)
    if snout_offset_px:
        port_disp = _offset_ports_display(port_disp, snout_offset_px)
    return port_disp, labels


# Default display offset between tracked head positions and port markers.
# The offset is applied only when ``offset_ports=True``.
DEFAULT_SNOUT_OFFSET_PX: float = 50.0


def _style_axis(ax: plt.Axes, label_size: int = 8, tick_size: int = 8) -> None:
    """Set tick and axis-label font sizes for a compact panel.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to restyle.
    label_size : int, optional
        Font size for the x and y axis labels.
    tick_size : int, optional
        Font size for the tick labels.
    """
    ax.tick_params(labelsize=tick_size)
    ax.xaxis.label.set_size(label_size)
    ax.yaxis.label.set_size(label_size)


def _set_equal_xy_limits(
    ax: plt.Axes,
    point_arrays: Sequence[np.ndarray],
    pad_frac: float = 0.08,
) -> None:
    """Set equal x/y data spans around all finite points.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes whose limits are set.
    point_arrays : sequence of np.ndarray
        Arrays with final dimension of length 2.
    pad_frac : float, optional
        Fractional padding added to the shared x/y span.
    """
    points = []
    for arr in point_arrays:
        arr = np.asarray(arr, dtype=float)
        if arr.size == 0:
            continue
        arr = arr.reshape(-1, 2)
        finite = np.isfinite(arr).all(axis=1)
        if np.any(finite):
            points.append(arr[finite])

    if not points:
        return

    xy = np.vstack(points)
    xmin, ymin = np.nanmin(xy, axis=0)
    xmax, ymax = np.nanmax(xy, axis=0)
    xmid = 0.5 * (xmin + xmax)
    ymid = 0.5 * (ymin + ymax)
    span = max(xmax - xmin, ymax - ymin)

    if span <= 0 or not np.isfinite(span):
        span = 1.0

    half = 0.5 * span * (1.0 + pad_frac)
    ax.set_xlim(xmid - half, xmid + half)
    ax.set_ylim(ymid - half, ymid + half)
    ax.set_aspect("equal", adjustable="box")


def _side_label(value: Any) -> str:
    """Return a side label while tolerating dict or sequence SIDE_LABELS."""
    try:
        return str(SIDE_LABELS[value])
    except Exception:
        return str(value)


def _outcome_label(error: Any) -> str:
    """Return a readable correct/error label."""
    if error is False or error == 0:
        return "correct"
    if error is True or error == 1:
        return "error"
    return str(error)


def _get_heading_direction(
    trial: Trial,
    transform: Optional[ViewTransform] = None,
    prefer_display_heading: bool = False,
) -> Optional[np.ndarray]:
    """Return heading direction in radians.

    Heading is read from ``heading_direction``, ``heading_dir``, or
    ``head_angle`` in that order. When ``prefer_display_heading`` is True, the
    angles are transformed to match the displayed trajectory orientation.

    Parameters
    ----------
    trial : dict
        Trial dictionary.
    transform : ViewTransform or None, optional
        Display transform used when converting to display heading.
    prefer_display_heading : bool, optional
        If True, convert raw-coordinate headings into display-coordinate
        headings.

    Returns
    -------
    np.ndarray or None
        Heading direction in radians, or None if no heading field is present.
    """
    if "heading_direction" in trial:
        heading = np.asarray(trial["heading_direction"], dtype=float)
    elif "heading_dir" in trial:
        heading = np.asarray(trial["heading_dir"], dtype=float)
    elif "head_angle" in trial:
        heading = np.asarray(trial["head_angle"], dtype=float)
    else:
        return None

    if prefer_display_heading and transform is not None:
        raw_vec = np.column_stack((np.cos(heading), np.sin(heading)))
        disp_vec = transform.apply(raw_vec)
        heading = np.arctan2(disp_vec[:, 1], disp_vec[:, 0])

    return heading


def _plot_heading_panel(
    ax: plt.Axes,
    trial: Trial,
    transform: ViewTransform,
    trace_color: str,
    event_styles: Dict[str, Tuple[str, str, str]],
    unwrap_heading: bool = True,
    prefer_display_heading: bool = False,
) -> None:
    """Draw heading direction over time.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes receiving the heading trace.
    trial : dict
        Trial dictionary.
    transform : ViewTransform
        Display transform.
    trace_color : str
        Color for the heading trace.
    event_styles : dict
        Event style dictionary mapping event names to ``(color, linestyle,
        label)`` tuples.
    unwrap_heading : bool, optional
        If True, unwrap circular heading values before plotting.
    prefer_display_heading : bool, optional
        If True, convert heading angles into display coordinates.
    """
    t = np.asarray(trial["t"], dtype=float)
    heading = _get_heading_direction(
        trial,
        transform=transform,
        prefer_display_heading=prefer_display_heading,
    )

    if heading is None:
        ax.text(0.5, 0.5, "heading direction not found",
                ha="center", va="center", transform=ax.transAxes, fontsize=9)
        ax.set_ylabel("heading\ndirection", fontsize=9)
        sns.despine(ax=ax)
        return

    if len(heading) != len(t):
        raise ValueError("heading direction and time arrays must have the same length.")

    if unwrap_heading:
        heading = np.unwrap(heading)

    ax.plot(t, heading, color=trace_color, linewidth=1.6, solid_capstyle="round")

    for name, (color, linestyle, _label) in event_styles.items():
        idx = _trial_event_index(trial, name)
        if idx is not None and 0 <= idx < len(t):
            ax.axvline(t[idx], color=color, linestyle=linestyle,
                       linewidth=1.2, alpha=0.9)

    ax.set_ylabel("heading\ndirection (rad)", fontsize=9)
    ax.yaxis.set_label_coords(-0.12, 0.5)
    sns.despine(ax=ax)


def _plot_velocity_panel(
    ax: plt.Axes,
    trial: Trial,
    key: str,
    ylabel: str,
    trace_color: str,
    event_styles: Dict[str, Tuple[str, str, str]],
    zero_line: bool = False,
) -> None:
    """Draw one velocity trace with shared event styling.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes receiving the velocity trace.
    trial : dict
        Trial dictionary.
    key : str
        Trial field containing the velocity array.
    ylabel : str
        Y-axis label.
    trace_color : str
        Color for the trace.
    event_styles : dict
        Event style dictionary mapping event names to ``(color, linestyle,
        label)`` tuples.
    zero_line : bool, optional
        If True, draw a horizontal line at zero.
    """
    t = np.asarray(trial["t"], dtype=float)

    if key not in trial:
        ax.text(0.5, 0.5, f"{key} not found",
                ha="center", va="center", transform=ax.transAxes, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        sns.despine(ax=ax)
        return

    y = np.asarray(trial[key], dtype=float)
    if len(y) != len(t):
        raise ValueError(f"{key} and time arrays must have the same length.")

    ax.plot(t, y, color=trace_color, linewidth=1.8, solid_capstyle="round")

    if zero_line:
        ax.axhline(0, color=OKABE_ITO["black"], linewidth=0.8,
                   linestyle=":", alpha=0.35)

    for name, (color, linestyle, _label) in event_styles.items():
        idx = _trial_event_index(trial, name)
        if idx is not None and 0 <= idx < len(t):
            ax.axvline(t[idx], color=color, linestyle=linestyle,
                       linewidth=1.2, alpha=0.9)

    ax.set_ylabel(ylabel, fontsize=9)
    ax.yaxis.set_label_coords(-0.12, 0.5)
    sns.despine(ax=ax)


def _draw_trajectory_panel(
    ax: plt.Axes,
    trial: Trial,
    transform: ViewTransform,
    port_disp: np.ndarray,
    port_marker: str = "s",
    port_size: float = 85.0,
    limit_arrays: Optional[Sequence[np.ndarray]] = None,
    pad_frac: float = 0.25,
) -> np.ndarray:
    """Draw one centroid trajectory with ports and a start marker.

    Both ``plot_trial`` and ``plot_trajectory_grid`` use this drawing routine so
    that paths are rendered consistently. The path is split at center entry into
    pre-entry and post-entry segments. Port and start markers are drawn on top.

    The caller passes ``port_disp`` already transformed, aligned, and offset, so
    this helper does not repeat those corrections. The caller also passes
    ``limit_arrays`` to control the view. A grid passes the full port geometry
    there, so every panel shares one centered frame rather than cropping to its
    own path.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes receiving the trajectory.
    trial : dict
        Trial dictionary.
    transform : ViewTransform
        Display transform applied to the centroid path.
    port_disp : np.ndarray
        Display-ready port coordinates, shape ``(n_ports, 2)``.
    port_marker : str, optional
        Marker for ports.
    port_size : float, optional
        Marker size for ports.
    limit_arrays : sequence of np.ndarray or None, optional
        Point arrays used to set the view limits. If None, the trajectory and
        the ports are used, which crops to the single trial. Pass a shared set
        to center every panel of a grid on the same frame.
    pad_frac : float, optional
        Fractional padding around the view limits.

    Returns
    -------
    np.ndarray
        The centroid path in display coordinates, shape ``(n, 2)``.
    """
    pre_color = OKABE_ITO["sky_blue"]
    post_color = OKABE_ITO["blue"]
    port_color = OKABE_ITO["black"]

    centroid = np.asarray(trial["centroid"], dtype=float)
    disp = transform.apply(centroid)
    n = len(disp)

    center_idx = _trial_event_index(trial, "center_entry")
    t = np.asarray(trial.get("t", []), dtype=float)
    if center_idx is None or not (0 <= center_idx < n):
        if t.size == n and np.any(np.isfinite(t)):
            center_idx = int(np.nanargmin(np.abs(t)))
        else:
            center_idx = 0

    if center_idx > 1:
        ax.plot(disp[:center_idx + 1, 0], disp[:center_idx + 1, 1],
                color=pre_color, linewidth=2.1, alpha=0.95,
                solid_capstyle="round", label="before center entry")
    if center_idx < n - 1:
        ax.plot(disp[center_idx:, 0], disp[center_idx:, 1],
                color=post_color, linewidth=2.1, alpha=0.95,
                solid_capstyle="round", label="after center entry")

    if len(port_disp):
        ax.scatter(port_disp[:, 0], port_disp[:, 1], s=port_size,
                   marker=port_marker, facecolors="white",
                   edgecolors=port_color, linewidths=1.3, zorder=5,
                   label="ports")

    ax.scatter(disp[0, 0], disp[0, 1], s=46, marker="o",
               facecolors=OKABE_ITO["black"], edgecolors="white",
               linewidths=0.8, zorder=6, label="start")

    arrays = limit_arrays if limit_arrays is not None else [disp, port_disp]
    _set_equal_xy_limits(ax, arrays, pad_frac=pad_frac)
    return disp


def _mark_trajectory_events(
    ax: plt.Axes,
    trial: Trial,
    disp: np.ndarray,
    n: int,
    mark_events: Sequence[str],
    marker_event_labels: Dict[str, str],
) -> None:
    """Mark named events as open circles on the trajectory panel.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Trajectory axes.
    trial : dict
        Trial dictionary.
    disp : np.ndarray
        Centroid path in display coordinates, shape ``(n, 2)``.
    n : int
        Number of samples in the path.
    mark_events : sequence of str
        Event names to mark.
    marker_event_labels : dict
        Mapping from event name to legend label.
    """
    for name in mark_events:
        idx = _trial_event_index(trial, name)
        if idx is not None and 0 <= idx < n:
            ax.scatter(disp[idx, 0], disp[idx, 1], s=46, marker="o",
                       facecolors="white",
                       edgecolors=EVENT_COLORS.get(name, OKABE_ITO["black"]),
                       linewidths=1.5, zorder=7,
                       label=marker_event_labels.get(
                           name, name.replace("_", " ")))


def plot_trial(
    trial: Trial,
    ports: Ports = None,
    transform: Optional[ViewTransform] = None,
    heading_every: int = 5,
    heading_len: float = 24.0,
    figsize: Tuple[float, float] = (8, 6),
    dpi: int = 140,
    show_heading: bool = True,
    mark_events: Sequence[str] = ("center_entry", "choice_entry",),
    port_marker: str = "s",
    port_size: float = 85.0,
    unwrap_heading: bool = True,
    prefer_display_heading: bool = False,
    align_ports: bool = True,
    offset_ports: bool = True,
    snout_offset_px: float = DEFAULT_SNOUT_OFFSET_PX,
) -> plt.Figure:
    """Plot one trial with trajectory, heading direction, and velocity panels.

    The left panel shows the centroid path in display coordinates, split into
    samples before and after center entry. The right panels show heading
    direction, linear velocity, and angular velocity over time.

    Parameters
    ----------
    trial : dict
        Trial dictionary from ``build_trials`` or ``load_session``.
    ports : np.ndarray, dict, or None, optional
        Port locations in raw arena coordinates. Arrays must have shape
        ``(n_ports, 2)``. Dictionaries map labels to ``(x, y)`` positions.
    transform : ViewTransform or None, optional
        Display transform. If None, ``PORTS_TOP`` is used.
    heading_every : int, optional
        Deprecated and ignored. Retained for backward compatibility with calls
        made before heading arrows were removed.
    heading_len : float, optional
        Deprecated and ignored. Retained for backward compatibility with calls
        made before heading arrows were removed.
    figsize : tuple of float, optional
        Figure size in inches.
    dpi : int, optional
        Figure resolution.
    show_heading : bool, optional
        Deprecated and ignored. Heading arrows are no longer drawn; the heading
        direction time-series panel remains available.
    mark_events : sequence of str, optional
        Events marked on the trajectory panel. The time-series panels always
        show center exit and choice entry as vertical lines.
    port_marker : str, optional
        Marker used for all ports.
    port_size : float, optional
        Marker size for ports.
    unwrap_heading : bool, optional
        If True, unwrap heading direction before plotting over time.
    prefer_display_heading : bool, optional
        If True, convert heading angles into display coordinates before
        plotting the heading direction trace.
    align_ports : bool, optional
        If True, force the response ports collinear and equally spaced for
        display. The correction affects plotted coordinates only.
    offset_ports : bool, optional
        If True, shift port markers away from the trajectory endpoints by
        ``snout_offset_px`` along the vertical display axis. This corrects for
        the tracked point stopping short of the port. Display only.
    snout_offset_px : float, optional
        Vertical shift used when ``offset_ports`` is True. Defaults to
        ``DEFAULT_SNOUT_OFFSET_PX``.

    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the trial plot.
    """
    if transform is None:
        transform = PORTS_TOP

    # Retained as no-op arguments so existing notebooks do not break in this
    # non-breaking plotting release.
    _ = heading_every, heading_len, show_heading

    pre_color = OKABE_ITO["sky_blue"]
    post_color = OKABE_ITO["blue"]
    trace_color = OKABE_ITO["blue"]
    port_color = OKABE_ITO["black"]

    event_styles = {
        "center_exit": (EVENT_COLORS["center_exit"], "--", "center exit"),
        "choice_entry": (EVENT_COLORS["choice_entry"], "--", "choice entry"),
    }

    fig = plt.figure(figsize=figsize, dpi=dpi)
    gs = fig.add_gridspec(
        3,
        2,
        width_ratios=(1.15, 1.0),
        hspace=0.30,
        wspace=0.30,
    )

    ax_traj = fig.add_subplot(gs[:, 0])
    ax_head = fig.add_subplot(gs[0, 1])
    ax_lin = fig.add_subplot(gs[1, 1], sharex=ax_head)
    ax_ang = fig.add_subplot(gs[2, 1], sharex=ax_head)

    centroid = np.asarray(trial["centroid"], dtype=float)
    disp = transform.apply(centroid)
    t = np.asarray(trial["t"], dtype=float)
    n = len(disp)

    if len(t) != n:
        raise ValueError("centroid and time arrays must have the same length.")

    port_disp, _port_labels = _display_ports(
        ports,
        transform,
        align=align_ports,
        snout_offset_px=snout_offset_px if offset_ports else 0.0,
    )

    disp = _draw_trajectory_panel(
        ax_traj, trial, transform, port_disp,
        port_marker=port_marker, port_size=port_size,
        pad_frac=0.25,
    )

    marker_event_labels = {
        "center_entry": "center entry",
        "center_exit": "center exit",
        "choice_entry": "choice entry",
    }

    if mark_events:
        _mark_trajectory_events(
            ax_traj, trial, disp, len(disp), mark_events, marker_event_labels
        )

    ax_traj.set_xlabel("display horizontal (px)")
    ax_traj.set_ylabel("display vertical, ports at top (px)")
    ax_traj.set_title("trajectory", fontsize=9)
    _style_axis(ax_traj)
    sns.despine(ax=ax_traj)

    _plot_heading_panel(
        ax_head,
        trial,
        transform,
        trace_color,
        event_styles,
        unwrap_heading=unwrap_heading,
        prefer_display_heading=prefer_display_heading,
    )

    _plot_velocity_panel(
        ax_lin,
        trial,
        "lin_vel",
        "linear\nvelocity (cm/s)",
        trace_color,
        event_styles,
        zero_line=False,
    )

    _plot_velocity_panel(
        ax_ang,
        trial,
        "ang_vel",
        "angular\nvelocity (rad/s)",
        trace_color,
        event_styles,
        zero_line=True,
    )

    for panel in (ax_head, ax_lin, ax_ang):
        _style_axis(panel)

    ax_head.set_title("heading and velocity", fontsize=9)
    ax_ang.set_xlabel("time from center entry (s)")
    plt.setp(ax_head.get_xticklabels(), visible=False)
    plt.setp(ax_lin.get_xticklabels(), visible=False)
    fig.align_ylabels([ax_head, ax_lin, ax_ang])

    try:
        cue = int(trial["cue"])
    except Exception:
        cue = trial.get("cue", "?")

    title = (f"trial {trial.get('idx', '?')}  cue {cue}  "
             f"target {_side_label(trial.get('target', '?'))}  "
             f"choice {_side_label(trial.get('choice', '?'))}  "
             f"{_outcome_label(trial.get('error', '?'))}")
    if "trial_type" in trial and trial["trial_type"] is not None:
        title = f"{title}  type {trial['trial_type']}"
    fig.suptitle(title, x=0.08, ha="left", fontsize=11)

    legend_handles = [
        plt.Line2D([0], [0], color=pre_color, lw=2.1,
                   label="before center entry"),
        plt.Line2D([0], [0], color=post_color, lw=2.1,
                   label="after center entry"),
        plt.Line2D([0], [0], marker="o", color="none",
                   markerfacecolor=OKABE_ITO["black"],
                   markeredgecolor="white", markersize=7, label="start"),
    ]

    if len(port_disp):
        legend_handles.append(
            plt.Line2D([0], [0], marker=port_marker, color="none",
                       markerfacecolor="white", markeredgecolor=port_color,
                       markersize=7, label="ports"))

    for name in mark_events:
        if name in EVENT_COLORS:
            legend_handles.append(
                plt.Line2D([0], [0], marker="o", color="none",
                           markerfacecolor="white",
                           markeredgecolor=EVENT_COLORS[name], markersize=7,
                           label=marker_event_labels[name]))

    for name, (color, linestyle, label) in event_styles.items():
        if _trial_event_index(trial, name) is not None:
            legend_handles.append(
                plt.Line2D([0], [0], color=color, lw=1.2,
                           ls=linestyle, label=label))

    fig.subplots_adjust(left=0.08, right=0.98, top=0.88, bottom=0.23)
    fig.legend(handles=legend_handles, loc="lower center",
               bbox_to_anchor=(0.5, 0.025),
               ncol=min(6, len(legend_handles)), frameon=False,
               fontsize=8, handlelength=2.0, columnspacing=1.2)

    return fig


PORT_STYLE: Dict[str, Tuple[str, str, str]] = {
    "center": ("s", OKABE_ITO["black"], "center"),
    "choice_L": ("s", OKABE_ITO["black"], "choice L"),
    "choice_R": ("s", OKABE_ITO["black"], "choice R"),
}


def _draw_ports(
    ax: plt.Axes,
    ports: Ports,
    transform: ViewTransform,
    size: float = 240,
    label: bool = True,
    align: bool = False,
    snout_offset_px: float = 0.0,
) -> None:
    """Draw port markers on an axes object.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes receiving the markers.
    ports : dict, np.ndarray, or None
        Port coordinates. Dictionaries map labels to points. Arrays must have
        shape ``(n_ports, 2)``.
    transform : ViewTransform
        Display transform applied before plotting.
    size : float, optional
        Marker size.
    label : bool, optional
        If True, add labels for the legend.
    align : bool, optional
        If True, force the response ports collinear and equally spaced before
        drawing. Display only.
    snout_offset_px : float, optional
        Vertical display shift applied to port markers. Zero disables the
        offset. Display only.
    """
    if ports is None:
        return
    prepared = _align_ports(ports) if align else ports
    if isinstance(prepared, dict):
        for key, pt in prepared.items():
            marker, col, name = PORT_STYLE.get(key, ("s", OKABE_ITO["black"], key))
            d = transform.apply(np.asarray(pt))
            d = _offset_ports_display(d.reshape(1, 2), snout_offset_px)[0]
            ax.scatter(d[0], d[1], s=size, marker=marker,
                       facecolors="white", edgecolors=col, linewidth=1.0,
                       zorder=2, label=name if label else None)
    else:
        d = transform.apply(np.asarray(prepared))
        d = _offset_ports_display(np.atleast_2d(d), snout_offset_px)
        ax.scatter(d[:, 0], d[:, 1], s=size, marker="s",
                   facecolors="white", edgecolors=OKABE_ITO["black"],
                   linewidth=1.0, zorder=2,
                   label="ports" if label else None)


def _vel_panel(
    ax: plt.Axes,
    trial: Trial,
    key: str,
    ylabel: str,
    color: Union[str, Tuple[float, float, float, float]],
    zero: bool = False,
) -> None:
    """Draw one velocity series with event markers."""
    event_styles = {
        "center_exit": (EVENT_COLORS["center_exit"], "--", "center exit"),
        "choice_entry": (EVENT_COLORS["choice_entry"], "--", "choice entry"),
    }
    _plot_velocity_panel(ax, trial, key, ylabel, color, event_styles, zero_line=zero)


def plot_trajectory_grid(
    trials: Sequence[Trial],
    ports: Ports = None,
    transform: ViewTransform = PORTS_TOP,
    n_cols: int = 6,
    trial_indices: Optional[Sequence[int]] = None,
    heading_every: int = 4,
    heading_len: float = 26.0,
    align_ports: bool = True,
    offset_ports: bool = True,
    snout_offset_px: float = DEFAULT_SNOUT_OFFSET_PX,
) -> plt.Figure:
    """Plot multiple trajectories as small panels.

    Each panel uses the trajectory drawing from ``plot_trial``. Paths are split
    at center entry, and axis labels are omitted to keep the panels compact.
    The ports and the trajectories are centered on the shared port geometry, so
    every panel uses one common frame rather than cropping to its own path.

    Parameters
    ----------
    trials : list of dict
        Trial dictionaries from ``build_trials`` or ``load_session``.
    ports : np.ndarray, dict, or None, optional
        Port locations in arena coordinates. Arrays must have shape
        ``(n_ports, 2)``. Dictionaries map port labels to ``(x, y)`` positions.
    transform : ViewTransform, optional
        Display transform applied to trajectories and ports.
    n_cols : int, optional
        Number of panels per row.
    trial_indices : sequence of int or None, optional
        Behavioral trial ``idx`` values to plot. If ``None``, all trials are
        plotted.
    heading_every : int, optional
        Deprecated and ignored. Retained for backward compatibility with calls
        made before heading arrows were removed.
    heading_len : float, optional
        Deprecated and ignored. Retained for backward compatibility with calls
        made before heading arrows were removed.
    align_ports : bool, optional
        If True, force the response ports collinear and equally spaced for
        display. Display only. Defaults to True.
    offset_ports : bool, optional
        If True, shift port markers away from the trajectory endpoints by
        ``snout_offset_px`` along the vertical display axis. Display only.
        Defaults to True.
    snout_offset_px : float, optional
        Vertical shift used when ``offset_ports`` is True. Defaults to
        ``DEFAULT_SNOUT_OFFSET_PX``.

    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the trajectory grid.
    """
    # Retained as no-op arguments so existing notebooks do not break in this
    # non-breaking plotting release.
    _ = heading_every, heading_len

    trials = _select_trials(trials, trial_indices)
    n = len(trials)
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.4 * n_cols, 2.4 * n_rows),
                             dpi=130)
    axes = np.atleast_1d(axes).ravel()

    port_disp, _port_labels = _display_ports(
        ports, transform, align=align_ports,
        snout_offset_px=snout_offset_px if offset_ports else 0.0)

    # Center every panel on one shared frame. The frame spans the ports and all
    # centroid paths, so panels are directly comparable and not cropped to each
    # single trial.
    all_paths = [transform.apply(np.asarray(t["centroid"], dtype=float))
                 for t in trials]
    limit_arrays = list(all_paths)
    if len(port_disp):
        limit_arrays = limit_arrays + [port_disp]

    for ax, trial in zip(axes, trials):
        _draw_trajectory_panel(
            ax, trial, transform, port_disp,
            port_marker="s", port_size=70.0,
            limit_arrays=limit_arrays, pad_frac=0.08)
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{trial['idx']} {SIDE_LABELS[trial['target']]}"
                     f"{'x' if trial['error'] else '+'}", fontsize=7)
        for sp in ax.spines.values():
            sp.set_color("#BDBDBD")

    for ax in axes[n:]:
        ax.set_visible(False)
    plt.tight_layout()
    return fig


def plot_kinematics_grid(
    trials: Sequence[Trial],
    kind: KinematicKind = "lin_vel",
    n_cols: int = 6,
    trial_indices: Optional[Sequence[int]] = None,
) -> plt.Figure:
    """Plot multiple velocity series as small panels.

    Parameters
    ----------
    trials : list of dict
        Trial dictionaries from ``build_trials`` or ``load_session``.
    kind : {'lin_vel', 'ang_vel'}, optional
        Trial field plotted on the y-axis.
    n_cols : int, optional
        Number of panels per row.
    trial_indices : sequence of int or None, optional
        Behavioral trial ``idx`` values to plot. If ``None``, all trials are
        plotted.

    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the velocity grid.
    """
    trials = _select_trials(trials, trial_indices)
    n = len(trials)
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.4 * n_cols, 1.9 * n_rows),
                             dpi=130, sharex=True)
    axes = np.atleast_1d(axes).ravel()
    # Y range covers the peak magnitude across displayed trials, plus 10 percent
    # headroom. Angular velocity is kept symmetric about zero.
    allv = np.concatenate([np.asarray(t[kind], dtype=float) for t in trials])
    finite = allv[np.isfinite(allv)]
    peak = float(np.nanmax(np.abs(finite))) if finite.size else 1.0
    hi = 1.10 * peak
    lo = -hi if kind == "ang_vel" else 0.0

    # X range tight to the span of the trial time vectors.
    t_min = min(float(np.nanmin(t["t"])) for t in trials)
    t_max = max(float(np.nanmax(t["t"])) for t in trials)

    for ax, trial in zip(axes, trials):
        ax.plot(
            trial["t"], trial[kind], color=OKABE_ITO["blue"], linewidth=1.4
        )
        if kind == "ang_vel":
            ax.axhline(0, color=OKABE_ITO["black"], linewidth=0.6, linestyle=":", alpha=0.35)
        event_lines = (
            ("center_exit", EVENT_COLORS["center_exit"]),
            ("choice_entry", EVENT_COLORS["choice_entry"]),
        )
        for name, c in event_lines:
            e = trial["events"][name]
            if 0 <= e < len(trial["t"]):
                ax.axvline(trial["t"][e], color=c, linewidth=1.0,
                           linestyle="--", alpha=0.8)
        ax.set_ylim(lo, hi)
        ax.set_xlim(t_min, t_max)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{trial['idx']} {SIDE_LABELS[trial['target']]}"
                     f"{'x' if trial['error'] else '+'}", fontsize=7)
        for sp in ax.spines.values():
            sp.set_color("#BDBDBD")

    for ax in axes[n:]:
        ax.set_visible(False)

    event_handles = [
        plt.Line2D([0], [0], color=EVENT_COLORS["center_exit"], lw=1.0,
                   ls="--", label="center exit"),
        plt.Line2D([0], [0], color=EVENT_COLORS["choice_entry"], lw=1.0,
                   ls="--", label="choice entry"),
    ]
    fig.legend(handles=event_handles, loc="upper left", frameon=False,
        bbox_to_anchor=(0, 1.1),
        fontsize=8, ncol=2, handlelength=2.0, columnspacing=1.2)
    plt.tight_layout()
    return fig


# ==============================================================================
# Figure-prepared trial views (centroid path and velocity series)
#
# These operate on trials passed through erpa.core.session.prepare_figure_trials,
# which adds the 'trajectory', 'velocity', and extrema fields they read.
# ==============================================================================

def plot_epoch_velocity(
    epoch_velocity: np.ndarray,
    window_before: int,
    framerate: int = DEFAULT_FRAMERATE,
    grouping: Optional[np.ndarray] = None,
    group_labels: Optional[Sequence[str]] = None,
    title: str = "Event-Aligned Velocity",
    ax: Optional[plt.Axes] = None
) -> plt.Axes:
    """Plot mean velocity profiles aligned to a behavioral event.

    The x-axis is time from the alignment event in milliseconds. If ``grouping``
    is provided, the mean and standard error are plotted separately for each
    group.

    Parameters
    ----------
    epoch_velocity : np.ndarray
        Event-aligned velocity array with shape ``(n_trials, window_size)``.
    window_before : int
        Number of frames before the event in each window. This defines the
        location of time zero.
    framerate : int, optional
        Video frame rate in frames per second.
    grouping : np.ndarray or None, optional
        Group label for each trial. If ``None``, all trials are averaged
        together.
    group_labels : list of str or None, optional
        Legend labels for group levels. Labels are indexed in the order returned
        by ``np.unique``.
    title : str, optional
        Axes title.
    ax : matplotlib.axes.Axes or None, optional
        Axes to plot on. If ``None``, a new figure and axes are created.

    Returns
    -------
    matplotlib.axes.Axes
        Axes containing the velocity plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5), dpi=150)

    window_size = epoch_velocity.shape[1]
    time_axis = (np.arange(window_size) - window_before) / framerate * 1000  # Convert to ms

    if grouping is None:
        # Plot all trials together
        mean_vel = np.nanmean(epoch_velocity, axis=0)
        sem_vel = np.nanstd(epoch_velocity, axis=0) / np.sqrt(
            np.sum(~np.isnan(epoch_velocity), axis=0)
        )

        ax.plot(time_axis, mean_vel, color=OKABE_ITO["blue"], linewidth=2)
        ax.fill_between(
            time_axis,
            mean_vel - sem_vel,
            mean_vel + sem_vel,
            color=OKABE_ITO["blue"],
            alpha=0.25,
        )
    else:
        # Plot by group
        unique_groups = np.unique(grouping[~np.isnan(grouping)])
        colors = [OKABE_CYCLE[i % len(OKABE_CYCLE)] for i in range(len(unique_groups))]

        for idx, group in enumerate(unique_groups):
            mask = grouping == group
            group_vel = epoch_velocity[mask]

            mean_vel = np.nanmean(group_vel, axis=0)
            sem_vel = np.nanstd(group_vel, axis=0) / np.sqrt(np.sum(~np.isnan(group_vel), axis=0))

            label = group_labels[idx] if group_labels else f"Group {int(group)}"
            ax.plot(time_axis, mean_vel, color=colors[idx], linewidth=2, label=label)
            ax.fill_between(time_axis, mean_vel - sem_vel, mean_vel + sem_vel,
                          color=colors[idx], alpha=0.3)

    # Add vertical line at event time
    ax.axvline(
        x=0,
        color=OKABE_ITO["black"],
        linestyle="--",
        linewidth=1,
        alpha=0.45,
        label="Event",
    )

    ax.set_xlabel('Time from event (ms)')
    ax.set_ylabel('Velocity (cm/s)')
    ax.set_title(title)
    ax.legend()
    sns.despine(ax=ax)

    return ax


def _view_limits(
    ax: plt.Axes,
    vt: ViewTransform,
    x0: float,
    x1: float,
    y0: float,
    y1: float,
) -> None:
    """Set axis limits after applying a display transform.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes whose limits are changed.
    vt : ViewTransform
        Display transform.
    x0, x1, y0, y1 : float
        Corners of the raw-coordinate limit box.
    """
    corners = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=float)
    d = vt.apply(corners)
    ax.set_xlim(d[:, 0].min(), d[:, 0].max())
    ax.set_ylim(d[:, 1].min(), d[:, 1].max())


def plot_trajectories(
    trials: Sequence[Trial],
    color_by: str = 'error',
    show_extrema: bool = False,
    show_full_arena: bool = False,
    max_trials_per_panel: int = 20,
    orientation: OrientationName = "paper",
    transform: Optional[ViewTransform] = None,
) -> plt.Figure:
    """Plot figure-prepared spatial trajectories.

    Requires the ``trajectory``, ``error``, ``localmin``, and ``lmax_idx``
    fields added by ``prepare_figure_trials``.

    Parameters
    ----------
    trials : list of dict
        Figure-prepared trials from ``prepare_figure_trials``.
    color_by : str, optional
        Variable used to color trajectories. Current behavior distinguishes
        outcomes when this is ``'error'`` and otherwise uses a neutral color.
    show_extrema : bool, optional
        If True, mark velocity maximum positions on each trajectory.
    show_full_arena : bool, optional
        If True, show the full arena. If False, zoom to the response area.
    max_trials_per_panel : int, optional
        Maximum number of trials drawn in each subplot.
    orientation : {'raw', 'paper'}, optional
        Display orientation. ``'raw'`` uses recorded arena coordinates.
        ``'paper'`` places the response ports across the top of the plot.
    transform : ViewTransform or None, optional
        Custom display transform. If not ``None``, it overrides ``orientation``.

    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the trajectory plots.
    """
    vt = _orientation_transform(orientation, transform=transform)

    colors = {
        'correct': OKABE_ITO['blue'],
        'error': OKABE_ITO['vermillion'],
        'neutral': OKABE_ITO['blue'],
    }

    n_panels = int(np.ceil(len(trials) / max_trials_per_panel))
    n_cols = min(n_panels, 5)
    n_rows = int(np.ceil(n_panels / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    if n_panels == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for panel_idx in range(n_panels):
        ax = axes[panel_idx]
        start_idx = panel_idx * max_trials_per_panel
        end_idx = min(start_idx + max_trials_per_panel, len(trials))

        for trial in trials[start_idx:end_idx]:
            traj = trial['trajectory']
            if traj.ndim == 3:
                traj = traj[:, :, 0]
            traj = vt.apply(traj)

            # Determine color
            if color_by == 'error':
                color = colors['correct'] if trial['error'] == 0 else colors['error']
            else:
                color = colors['neutral']

            ax.plot(traj[:, 0], traj[:, 1], c=color, alpha=0.5, linewidth=0.8)

            if show_extrema:
                max_idx = trial['lmax_idx'] or 0
                if max_idx > 0:
                    ax.scatter(traj[max_idx, 0], traj[max_idx, 1],
                              c=OKABE_ITO['orange'], s=20, zorder=2)

        if show_full_arena:
            _view_limits(ax, vt, 0, 1006, 0, 758)
        else:
            _view_limits(ax, vt, 700, 1006, 150, 600)

    # Hide empty panels
    for ax in axes[n_panels:]:
        ax.set_visible(False)

    plt.tight_layout()
    return fig


def plot_velocity_profiles(
    trials: Sequence[Trial],
    color_by: str = 'error',
    show_extrema: bool = False,
    max_trials_per_panel: int = 20
) -> plt.Figure:
    """Plot figure-prepared velocity time series.

    Requires the ``velocity``, ``error``, ``localmin``, and ``lmax_idx``
    fields added by ``prepare_figure_trials``.

    Parameters
    ----------
    trials : list of dict
        Figure-prepared trials from ``prepare_figure_trials``.
    color_by : str, optional
        Variable used to color traces. Current behavior distinguishes outcomes
        when this is ``'error'`` and otherwise uses the correct-trial color.
    show_extrema : bool, optional
        If True, mark the maximum velocity.
    max_trials_per_panel : int, optional
        Maximum number of trials drawn in each subplot.

    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the velocity plots.
    """
    colors = {
        'correct': OKABE_ITO['blue'],
        'error': OKABE_ITO['vermillion'],
    }

    n_panels = int(np.ceil(len(trials) / max_trials_per_panel))
    n_cols = min(n_panels, 5)
    n_rows = int(np.ceil(n_panels / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    if n_panels == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for panel_idx in range(n_panels):
        ax = axes[panel_idx]
        start_idx = panel_idx * max_trials_per_panel
        end_idx = min(start_idx + max_trials_per_panel, len(trials))

        for trial in trials[start_idx:end_idx]:
            vel = trial['velocity']
            time_axis = np.arange(len(vel))

            if color_by == 'error':
                color = colors['correct'] if trial['error'] == 0 else colors['error']
            else:
                color = colors['correct']

            ax.plot(time_axis, vel, c=color, alpha=0.5, linewidth=0.8)

            if show_extrema:
                max_idx = trial['lmax_idx']
                if max_idx is not None:
                    ax.scatter(max_idx, vel[max_idx], c=OKABE_ITO['orange'], s=30, zorder=2)

        ax.set_xlabel('Frame')
        ax.set_ylabel('Velocity (cm/s)')

    for ax in axes[n_panels:]:
        ax.set_visible(False)

    plt.tight_layout()
    return fig
