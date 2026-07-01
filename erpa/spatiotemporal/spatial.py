"""
Geometric and kinematic measures for ERPA trials.

This module computes scalar and curve-based measures from the decision movement,
defined as the segment from center exit to choice entry. The functions use a
one-dimensional decision axis for the collinear two-choice port layout.
Distances are reported in centimeters when a pixel-to-centimeter scale is
available.
"""

import numpy as np
import pandas as pd
import warnings
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple
from numpy.typing import ArrayLike
from scipy.ndimage import gaussian_filter1d

from erpa.util import as_meta_frame


Trial = Dict[str, Any]
Ports = Mapping[str, ArrayLike]


def decision_axis(ports: Ports) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Compute the decision axis from port coordinates.

    The decision axis is the unit vector from the left choice port to the right
    choice port. Choice ``1`` is on the positive side of this axis and choice
    ``0`` is on the negative side.

    Parameters
    ----------
    ports : dict
        Port locations. Required keys are ``"center"``, ``"choice_L"``, and
        ``"choice_R"``. Each value must contain an ``(x, y)`` coordinate.

    Returns
    -------
    center : numpy.ndarray
        Center-port coordinate with shape ``(2,)``.
    u : numpy.ndarray
        Unit vector pointing from the left choice port to the right choice port,
        with shape ``(2,)``.
    dR : float
        Signed distance, in pixels, from the center port to the right choice
        port along ``u``.
    dL : float
        Signed distance, in pixels, from the center port to the left choice port
        along ``u``.

    Notes
    -----
    ``dR`` is expected to be positive and ``dL`` negative when the center port
    lies between the two choice ports.
    """
    center = np.asarray(ports["center"], float)
    cL = np.asarray(ports["choice_L"], float)
    cR = np.asarray(ports["choice_R"], float)
    axis = cR - cL
    u = axis / np.linalg.norm(axis)
    dR = float((cR - center) @ u)   # > 0
    dL = float((cL - center) @ u)   # < 0
    return center, u, dR, dL


def cm_per_pixel(ports: Ports, port_spacing_cm: float = 4.0) -> float:
    """
    Derive the centimeter-per-pixel scale from the choice-port spacing.

    The left and right choice ports are assumed to be separated by two equal
    port gaps. The physical distance between adjacent ports is
    ``port_spacing_cm``.

    Parameters
    ----------
    ports : dict
        Port locations. Required keys are ``"choice_L"`` and ``"choice_R"``.
        Each value must contain an ``(x, y)`` coordinate.
    port_spacing_cm : float, optional
        Physical distance between adjacent ports in centimeters.

    Returns
    -------
    float
        Centimeters per pixel.

    Raises
    ------
    ValueError
        If the detected left and right choice ports have the same pixel
        coordinate.
    """
    cL = np.asarray(ports["choice_L"], float)
    cR = np.asarray(ports["choice_R"], float)
    pix_dist = float(np.linalg.norm(cR - cL))   # left to right, two gaps
    if pix_dist <= 0:
        raise ValueError("choice ports coincide; cannot derive a pixel scale.")
    return (2.0 * port_spacing_cm) / pix_dist


def _smooth_xy(xy: np.ndarray, sigma: float = 1.5) -> np.ndarray:
    """
    Smooth a two-dimensional trajectory with a Gaussian filter.

    Parameters
    ----------
    xy : numpy.ndarray
        Position array with shape ``(n_frames, 2)``.
    sigma : float, optional
        Standard deviation of the Gaussian kernel, in frames.

    Returns
    -------
    numpy.ndarray
        Smoothed trajectory with shape ``(n_frames, 2)``. Trajectories with
        fewer than three frames are returned unchanged.
    """
    if xy.shape[0] < 3:
        return xy
    return np.column_stack([gaussian_filter1d(xy[:, 0], sigma),
                            gaussian_filter1d(xy[:, 1], sigma)])


def _segment_xy(
    trial: Trial,
    ce: int,
    ch: int,
    node: str,
    node_names: Optional[Sequence[str]],
) -> Optional[np.ndarray]:
    """
    Extract node positions for the decision segment.

    Parameters
    ----------
    trial : dict
        ERPA trial dictionary.
    ce : int
        Center-exit frame index within the trial window.
    ch : int
        Choice-entry frame index within the trial window.
    node : str
        ``"centroid"`` or a pose-node name.
    node_names : sequence of str or None
        Pose-node names in the order used by ``trial["nodes"]``.

    Returns
    -------
    numpy.ndarray or None
        Position array with shape ``(n_frames, 2)`` for frames ``ce`` through
        ``ch`` inclusive. Returns ``None`` when ``node`` is not available.
    """
    if node == "centroid":
        return trial["centroid"][ce:ch + 1].astype(float)
    if node_names is None or node not in list(node_names):
        return None
    j = list(node_names).index(node)
    return trial["nodes"][ce:ch + 1, j, :].astype(float)


def _arclen(xy: np.ndarray) -> np.ndarray:
    """
    Compute cumulative arc length for a two-dimensional trajectory.

    Parameters
    ----------
    xy : numpy.ndarray
        Position array with shape ``(n_frames, 2)``.

    Returns
    -------
    numpy.ndarray
        Cumulative path length in the same units as ``xy`` and with shape
        ``(n_frames,)``.
    """
    step = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(step)])


def arclength_resample(xy: ArrayLike, n_points: int = 50) -> np.ndarray:
    """
    Resample a two-dimensional trajectory at equal arc-length intervals.

    Arc-length resampling represents path shape as a function of distance
    traveled rather than elapsed time.

    Parameters
    ----------
    xy : numpy.ndarray
        Position array with shape ``(n_frames, 2)``.
    n_points : int, optional
        Number of samples in the resampled trajectory.

    Returns
    -------
    numpy.ndarray
        Resampled trajectory with shape ``(n_points, 2)``. If the original path
        length is zero, the first point is repeated.
    """
    xy = np.asarray(xy, float)
    arc = _arclen(xy)
    if arc[-1] <= 0:
        return np.repeat(xy[:1], n_points, axis=0)
    grid = np.linspace(0, arc[-1], n_points)
    return np.column_stack([np.interp(grid, arc, xy[:, 0]),
                            np.interp(grid, arc, xy[:, 1])])


@dataclass
class TrialGeometry:
    """
    Geometry and kinematic landmarks for one trial.

    Attributes
    ----------
    idx : int
        Trial index after filtering.
    choice : int
        Chosen side coded as ``0`` for left and ``1`` for right.
    error : int
        Trial outcome coded as ``1`` for error and ``0`` for correct.
    s : float
        Sign of the chosen direction along the decision axis.
    fps : float
        Video frame rate in frames per second.
    pix : float
        Centimeters per pixel.
    center : numpy.ndarray
        Center-port coordinate.
    u : numpy.ndarray
        Unit decision-axis vector.
    chosen_port : numpy.ndarray
        Coordinate of the chosen port.
    unchosen_port : numpy.ndarray
        Coordinate of the unchosen port.
    d_chosen : float
        Center-to-chosen-port distance in pixels.
    ce : int
        Center-exit frame index.
    ch : int
        Choice-entry frame index.
    xy : numpy.ndarray
        Raw decision-segment positions.
    sm : numpy.ndarray
        Smoothed decision-segment positions.
    arc : numpy.ndarray
        Cumulative arc length of the smoothed path.
    total : float
        Total smoothed path length in pixels.
    disp : float
        Straight-line displacement of the smoothed path in pixels.
    p : numpy.ndarray
        Signed projection of the path onto the decision axis.
    i_dev : int
        Index of maximum excursion toward the unchosen side.
    dev_pix : float
        Maximum excursion toward the unchosen side in pixels.
    dev_point : numpy.ndarray
        Position at ``i_dev``.
    idx_commit : int
        First index where path progress reaches ``commit_frac`` of the
        center-to-chosen-port distance.
    commit_thr : float
        Commitment threshold in pixels.
    commit_frac : float
        Fraction of the center-to-chosen-port distance used for the commitment
        threshold.
    commit_point : numpy.ndarray
        Position at ``idx_commit``.
    i_land : int
        Index of the heading landmark.
    head_point : numpy.ndarray
        Position at ``i_land``.
    head_vec : numpy.ndarray
        Unit heading vector at ``i_land``.
    head_to_chosen : float
        Projection of heading onto the chosen direction. Values near ``1`` point
        toward the chosen port; values near ``-1`` point toward the unchosen
        port.
    landmark_frac : float
        Fraction of path length used to select the heading landmark.
    t : numpy.ndarray
        Trial time axis.
    lin_vel : numpy.ndarray
        Linear velocity array.
    ang_vel : numpy.ndarray
        Angular velocity array.
    i_pkv : int
        Index of peak linear velocity within the decision segment.
    i_pka : int
        Index of peak absolute angular velocity near ``i_pkv``.
    win : tuple
        Start and stop indices of the angular-velocity search window.
    half : int
        Half-width of the angular-velocity search window in frames.
    scalars : dict
        Scalar measures for the trial.
    """
    idx: int
    choice: int
    error: int
    s: float
    fps: float
    pix: float
    center: np.ndarray
    u: np.ndarray
    chosen_port: np.ndarray
    unchosen_port: np.ndarray
    d_chosen: float
    ce: int
    ch: int
    xy: np.ndarray
    sm: np.ndarray
    arc: np.ndarray
    total: float
    disp: float
    p: np.ndarray
    i_dev: int
    dev_pix: float
    dev_point: np.ndarray
    idx_commit: int
    commit_thr: float
    commit_frac: float
    commit_point: np.ndarray
    i_land: int
    head_point: np.ndarray
    head_vec: np.ndarray
    head_to_chosen: float
    landmark_frac: float
    t: np.ndarray
    lin_vel: np.ndarray
    ang_vel: np.ndarray
    i_pkv: int
    i_pka: int
    win: Tuple[int, int]
    half: int
    scalars: Dict[str, Any]


def compute_trial_geometry(
    trial: Trial,
    ports: Ports,
    node: str = "centroid",
    node_names: Optional[Sequence[str]] = None,
    pix: Optional[float] = None,
    port_spacing_cm: float = 4.0,
    landmark_frac: float = 0.2,
    commit_frac: float = 0.5,
    smooth_sigma: float = 1.5,
) -> Optional[Tuple[Dict[str, Any], TrialGeometry]]:
    """
    Compute scalar measures and geometry landmarks for one trial.

    The decision segment runs from center exit through choice entry. The function
    returns ``None`` when that segment is clipped, has fewer than four position
    samples, contains non-finite positions, or uses an unavailable node.

    Parameters
    ----------
    trial : dict
        ERPA trial dictionary from ``build_trials`` or ``load_session``.
    ports : dict
        Port locations with keys ``"center"``, ``"choice_L"``, and
        ``"choice_R"``.
    node : str, optional
        Position source used for path measures. Use ``"centroid"`` for body
        centroid or a pose-node name for a tracked keypoint.
    node_names : sequence of str or None, optional
        Pose-node names used when ``node`` selects a keypoint from
        ``trial["nodes"]``.
    pix : float or None, optional
        Centimeters per pixel. If ``None``, the value is derived from
        ``ports`` and ``port_spacing_cm``.
    port_spacing_cm : float, optional
        Physical distance between adjacent ports in centimeters.
    landmark_frac : float, optional
        Fraction of path length used to select the early-heading landmark.
    commit_frac : float, optional
        Fraction of the center-to-chosen-port distance used to define the
        commitment threshold.
    smooth_sigma : float, optional
        Standard deviation of the Gaussian filter used for path length and
        tortuosity.

    Returns
    -------
    tuple of dict and TrialGeometry, or None
        The first element is a scalar-measure dictionary. The second element is a
        ``TrialGeometry`` object containing landmarks used for plotting and
        verification. Returns ``None`` when the trial cannot be measured.

    Notes
    -----
    ``dev_unchosen_cm`` is positive when the path moves toward the unchosen port.
    ``head_to_chosen`` is the heading projection onto the chosen direction at the
    selected path-length landmark.
    """
    if pix is None:
        pix = cm_per_pixel(ports, port_spacing_cm)
    center, u, dR, dL = decision_axis(ports)
    t = trial
    ce = t["events"]["center_exit"]
    ch = t["events"]["choice_entry"]
    if ce < 1 or ch >= len(t["lin_vel"]) or ch <= ce:        # clipped segment
        return None

    xy = _segment_xy(t, ce, ch, node, node_names)
    if xy is None or xy.shape[0] < 4 or not np.all(np.isfinite(xy)):
        return None

    choice = int(t["choice"])
    s = 1.0 if choice == 1 else -1.0                          # chosen dir along u
    d_chosen = abs(dR if choice == 1 else dL)                 # center to chosen, px
    chosen_port = np.asarray(ports["choice_R"] if choice == 1
                             else ports["choice_L"], float)
    unchosen_port = np.asarray(ports["choice_L"] if choice == 1
                               else ports["choice_R"], float)

    p = (xy - center) @ u                                     # signed pos on axis
    toward_unchosen = -s * p                                  # + = wrong direction
    i_dev = int(np.argmax(toward_unchosen))
    dev_pix = float(toward_unchosen[i_dev])

    sm = _smooth_xy(xy, smooth_sigma)
    step = np.linalg.norm(np.diff(sm, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(step)])
    total = arc[-1]
    disp = np.linalg.norm(sm[-1] - sm[0])

    head = t["head_angle"][ce:ch + 1]
    i_land = int(np.searchsorted(arc, landmark_frac * total))
    i_land = min(i_land, len(head) - 1)
    hv = np.array([np.cos(head[i_land]), np.sin(head[i_land])])
    head_to_chosen = float(s * (hv @ u))

    prog = s * p                                              # progress to chosen
    thr = commit_frac * d_chosen
    idx_commit = int(np.argmax(prog >= thr)) if np.any(prog >= thr) else len(prog) - 1

    fps = t["framerate"]
    lv = np.asarray(t["lin_vel"], float)
    av = np.asarray(t["ang_vel"], float)

    # Peak linear velocity during the decision movement, from center exit
    # through choice entry. ``lin_vel`` is already in cm/s when trials are built
    # from ``compute_session_series``.
    lv_seg = lv[ce:ch + 1]
    if lv_seg.size and np.any(np.isfinite(lv_seg)):
        i_pkv = ce + int(np.nanargmax(lv_seg))
        peak_lin_vel = float(lv[i_pkv])
    else:
        i_pkv = ch
        peak_lin_vel = np.nan

    half = int(round(0.12 * fps))
    a0 = max(0, i_pkv - half)
    a1 = min(len(av), i_pkv + half + 1)
    win = av[a0:a1]
    if len(win):
        i_pka = a0 + int(np.argmax(np.abs(win)))
        peak_ang_vel = float(av[i_pka])
        ang_peak_lag = float((i_pka - i_pkv) / fps)
    else:
        i_pka = i_pkv
        peak_ang_vel, ang_peak_lag = np.nan, np.nan

    # Order preserved exactly, so the measure table is unchanged.
    scalars = dict(
        idx=t["idx"], absolute_trial=int(t.get("absolute_trial", -1)),
        target=t["target"], choice=choice,
        trial_type=t.get("trial_type", np.nan),
        error=int(t["error"]),
        cue=t.get("cue", np.nan), sampling=t["sampling"], rt=t["rt"],
        vel_choice=float(lv[ch]),
        peak_lin_vel=peak_lin_vel,
        peak_ang_vel=peak_ang_vel,
        ang_peak_lag=ang_peak_lag,
        t_choice=(ch - ce) / fps,
        t_commit=int(idx_commit) / fps,
        dev_unchosen_cm=dev_pix * pix,
        dev_unchosen_norm=dev_pix / d_chosen,
        t_dev=i_dev / fps,
        head_to_chosen=head_to_chosen,
        tortuosity=float(total / disp) if disp > 0 else np.nan,
        path_len_cm=total * pix,
        path_len_norm=total / d_chosen,
    )

    geom = TrialGeometry(
        idx=int(t["idx"]), choice=choice, error=int(t["error"]), s=s,
        fps=float(fps), pix=float(pix),
        center=center, u=u, chosen_port=chosen_port, unchosen_port=unchosen_port,
        d_chosen=float(d_chosen),
        ce=int(ce), ch=int(ch), xy=xy, sm=sm, arc=arc,
        total=float(total), disp=float(disp), p=p,
        i_dev=i_dev, dev_pix=dev_pix, dev_point=xy[i_dev],
        idx_commit=int(idx_commit), commit_thr=float(thr),
        commit_frac=float(commit_frac),
        commit_point=xy[min(idx_commit, len(xy) - 1)],
        i_land=i_land, head_point=xy[i_land], head_vec=hv,
        head_to_chosen=head_to_chosen, landmark_frac=float(landmark_frac),
        t=np.asarray(t["t"], float), lin_vel=np.asarray(lv, float),
        ang_vel=np.asarray(av, float),
        i_pkv=int(i_pkv), i_pka=int(i_pka), win=(int(a0), int(a1)), half=int(half),
        scalars=scalars,
    )
    return scalars, geom


def scalar_feature_matrix(
    trials: Sequence[Trial],
    ports: Ports,
    node: str = "centroid",
    node_names: Optional[Sequence[str]] = None,
    pix: Optional[float] = None,
    port_spacing_cm: float = 4.0,
    landmark_frac: float = 0.2,
    commit_frac: float = 0.5,
    smooth_sigma: float = 1.5,
) -> pd.DataFrame:
    """
    Build one row of scalar features for each measurable trial.

    Parameters
    ----------
    trials : list of dict
        Trial dictionaries from ``build_trials`` or ``load_session``.
    ports : dict
        Port locations with keys ``"center"``, ``"choice_L"``, and
        ``"choice_R"``.
    node : str, optional
        Position source used for path measures. Use ``"centroid"`` for body
        centroid or a pose-node name.
    node_names : sequence of str or None, optional
        Pose-node names used when ``node`` selects a keypoint from
        ``trial["nodes"]``.
    pix : float or None, optional
        Centimeters per pixel. If ``None``, the value is derived from the port
        layout.
    port_spacing_cm : float, optional
        Physical distance between adjacent ports in centimeters.
    landmark_frac : float, optional
        Fraction of path length used to select the heading landmark.
    commit_frac : float, optional
        Fraction of the center-to-chosen-port distance used to define commitment.
    smooth_sigma : float, optional
        Standard deviation of the Gaussian filter used for path length and
        tortuosity.

    Returns
    -------
    pandas.DataFrame
        Feature table with one row per measured trial.

    Notes
    -----
    The table includes behavioral measures such as ``sampling`` and ``RT``,
    kinematic measures such as ``vel_choice``, ``peak_lin_vel``, and
    ``peak_ang_vel``, and path measures such as ``dev_unchosen_cm`` and
    ``tortuosity``. Trials with clipped or invalid decision segments are skipped.
    """
    if pix is None:
        pix = cm_per_pixel(ports, port_spacing_cm)
    rows, n_skip = [], 0

    for t in trials:
        res = compute_trial_geometry(
            t, ports, node=node, node_names=node_names, pix=pix,
            port_spacing_cm=port_spacing_cm, landmark_frac=landmark_frac,
            commit_frac=commit_frac, smooth_sigma=smooth_sigma)
        if res is None:
            n_skip += 1
            continue
        rows.append(res[0])

    if n_skip:
        warnings.warn(f"scalar_feature_matrix skipped {n_skip} trials with a "
                      f"clipped or invalid decision segment.")
    return pd.DataFrame(rows)


def deviation_curves(
    trials: Sequence[Trial],
    ports: Ports,
    n_points: int = 50,
    node: str = "centroid",
    node_names: Optional[Sequence[str]] = None,
    normalize: bool = True,
    pix: Optional[float] = None,
    port_spacing_cm: float = 4.0,
    toward: str = "chosen",
    smooth_sigma: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Build decision-axis deviation curves over normalized path length.

    Each trajectory is projected onto the decision axis and resampled at equal
    arc-length intervals. The output x-axis is fraction of path traveled, not
    time.

    Parameters
    ----------
    trials : list of dict
        Trial dictionaries from ``build_trials`` or ``load_session``.
    ports : dict
        Port locations with keys ``"center"``, ``"choice_L"``, and
        ``"choice_R"``.
    n_points : int, optional
        Number of samples in each resampled curve.
    node : str, optional
        Position source used for the path. Use ``"centroid"`` for body centroid
        or a pose-node name.
    node_names : sequence of str or None, optional
        Pose-node names used when ``node`` selects a keypoint from
        ``trial["nodes"]``.
    normalize : bool, optional
        If ``True``, divide projected position by the center-to-chosen-port
        distance. If ``False``, return centimeters when ``pix`` is available.
    pix : float or None, optional
        Centimeters per pixel. Used only when ``normalize=False``. If ``None``,
        the value is derived from the port layout.
    port_spacing_cm : float, optional
        Physical distance between adjacent ports in centimeters.
    toward : {"chosen", "unchosen"}, optional
        Direction used for the sign convention. ``"chosen"`` makes movement
        toward the chosen port positive. ``"unchosen"`` makes movement toward the
        unchosen port positive.
    smooth_sigma : float, optional
        Standard deviation of the Gaussian filter applied before arc-length
        resampling.

    Returns
    -------
    grid : numpy.ndarray
        Normalized arc-length grid with shape ``(n_points,)``.
    Y : numpy.ndarray
        Deviation curves with shape ``(n_kept, n_points)``.
    meta : pandas.DataFrame
        Trial metadata for the kept curves.

    Notes
    -----
    With ``toward="chosen"``, movement toward the chosen port is positive and
    movement toward the unchosen port is negative. With ``toward="unchosen"``,
    that sign convention is reversed to match ``dev_unchosen_cm``.
    """
    center, u, dR, dL = decision_axis(ports)
    if not normalize and pix is None:
        pix = cm_per_pixel(ports, port_spacing_cm)
    grid = np.linspace(0, 1, n_points)
    sgn = 1.0 if toward == "chosen" else -1.0
    rows, meta, n_skip = [], [], 0

    for t in trials:
        ce = t["events"]["center_exit"]
        ch = t["events"]["choice_entry"]
        if ce < 1 or ch >= len(t["lin_vel"]) or ch <= ce:
            n_skip += 1
            continue
        xy = _segment_xy(t, ce, ch, node, node_names)
        if xy is None or xy.shape[0] < 4 or not np.all(np.isfinite(xy)):
            n_skip += 1
            continue
        xy = _smooth_xy(xy, smooth_sigma)

        choice = int(t["choice"])
        s = 1.0 if choice == 1 else -1.0
        d_chosen = abs(dR if choice == 1 else dL)
        q = sgn * s * ((xy - center) @ u)     # pixels, toward chosen or unchosen
        if normalize:
            q = q / d_chosen                  # dimensionless fraction of the
        elif pix is not None:                 # center-to-port distance
            q = q * pix                       # centimeters

        arc = _arclen(xy)
        if arc[-1] <= 0:
            n_skip += 1
            continue
        rows.append(np.interp(grid * arc[-1], arc, q))
        meta.append({k: t[k] for k in ("idx", "target", "choice",
                                       "cue", "rt", "error") if k in t})

    if n_skip:
        warnings.warn(f"deviation_curves skipped {n_skip} trials with a clipped "
                      f"or invalid decision segment.")
    return grid, np.array(rows), as_meta_frame(meta)


def arclength_signal(
    trials: Sequence[Trial],
    key: str = "lin_vel",
    n_points: int = 50,
    node: str = "centroid",
    node_names: Optional[Sequence[str]] = None,
    smooth_sigma: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Resample a per-frame signal over normalized path length.

    This represents a signal as a function of distance traveled along the
    decision path rather than as a function of time.

    Parameters
    ----------
    trials : list of dict
        Trial dictionaries from ``build_trials`` or ``load_session``.
    key : str, optional
        Trial signal to resample.
    n_points : int, optional
        Number of samples in each resampled signal.
    node : str, optional
        Position source used to compute path length. Use ``"centroid"`` for body
        centroid or a pose-node name.
    node_names : sequence of str or None, optional
        Pose-node names used when ``node`` selects a keypoint from
        ``trial["nodes"]``.
    smooth_sigma : float, optional
        Standard deviation of the Gaussian filter applied before arc-length
        calculation.

    Returns
    -------
    grid : numpy.ndarray
        Normalized arc-length grid with shape ``(n_points,)``.
    Y : numpy.ndarray
        Resampled signals with shape ``(n_kept, n_points)``.
    meta : pandas.DataFrame
        Trial metadata for the kept curves.
    """
    grid = np.linspace(0, 1, n_points)
    rows, meta, n_skip = [], [], 0
    for t in trials:
        ce = t["events"]["center_exit"]
        ch = t["events"]["choice_entry"]
        if ce < 1 or ch >= len(t[key]) or ch <= ce:
            n_skip += 1
            continue
        xy = _segment_xy(t, ce, ch, node, node_names)
        if xy is None or xy.shape[0] < 4 or not np.all(np.isfinite(xy)):
            n_skip += 1
            continue
        xy = _smooth_xy(xy, smooth_sigma)
        sig = np.asarray(t[key][ce:ch + 1], float)
        if sig.shape[0] != xy.shape[0] or not np.all(np.isfinite(sig)):
            n_skip += 1
            continue
        arc = _arclen(xy)
        if arc[-1] <= 0:
            n_skip += 1
            continue
        rows.append(np.interp(grid * arc[-1], arc, sig))
        meta.append({k: t[k] for k in ("idx", "target", "choice",
                                       "cue", "rt", "error") if k in t})
    if n_skip:
        warnings.warn(f"arclength_signal skipped {n_skip} trials.")
    return grid, np.array(rows), as_meta_frame(meta)
