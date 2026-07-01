"""Functions for storing registered FDA results.

This module converts registered trial curves and FDA scores into labeled
xarray datasets. It also includes functions for saving session datasets,
loading pooled datasets, and computing mean curves by treatment.

Notes
-----
These functions require xarray and a netCDF backend. Install the optional
storage dependencies before using this module.
"""

from os import PathLike
from typing import Any, Dict, Sequence, Union, TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import xarray as xr

from erpa.spatiotemporal.curves import movement_curves
from erpa.util import as_meta_frame


def register_to_dataset(
    trials: Sequence[Dict[str, Any]],
    key: str = "lin_vel",
    n_points: int = 50,
    pad: int = 2,
    treatment: str = "unknown",
    session_id: str = "session",
    n_comp: int = 3,
    max_iter: int = 8,
) -> "xr.Dataset":
    """Register movement curves and return them as an xarray Dataset.

    Parameters
    ----------
    trials : list of dict
        Trial dictionaries from ``erpa.core.session.load_session`` or
        ``erpa.core.session.build_trials``.
    key : str, optional
        Trial series to register, such as ``"lin_vel"`` or ``"ang_vel"``.
    n_points : int, optional
        Number of points in each resampled movement curve.
    pad : int, optional
        Number of points used to pad each movement curve before registration.
    treatment : str, optional
        Session treatment label, such as ``"pbs"`` or ``"muscimol"``.
    session_id : str, optional
        Recording or session identifier.
    n_comp : int, optional
        Number of fPCA components to compute for amplitude, phase, and joint
        scores.
    max_iter : int, optional
        Maximum number of iterations used by the elastic registration
        functions.

    Returns
    -------
    xarray.Dataset
        Dataset containing raw curves, registered curves, warping functions,
        the template curve, fPCA scores, and trial-level coordinates.

    Notes
    -----
    The returned dataset contains these data variables:

    raw : (trial, time)
        Resampled movement curves before registration.
    registered : (trial, time)
        Amplitude curves after registration.
    warping : (trial, time)
        Warping functions from elastic registration.
    template : (time,)
        Karcher mean curve.
    amp_scores : (trial, comp)
        Amplitude fPCA scores.
    phase_scores : (trial, comp)
        Phase fPCA scores.
    joint_scores : (trial, comp)
        Joint fPCA scores.

    Trial coordinates include ``trial_idx``, ``target``, ``error``,
    ``cue``, ``rt``, ``treatment``, and ``session_id``.
    """
    import xarray as xr

    from erpa.fda.registration import elastic_register, amplitude_phase_features

    grid, Y, meta = movement_curves(trials, key=key, n_points=n_points, pad=pad)
    meta = as_meta_frame(meta)
    fn, gam, template = elastic_register(grid, Y, max_iter=max_iter)
    feats = amplitude_phase_features(grid, Y, n_comp=n_comp, max_iter=max_iter)

    # Per-trial conditions, aligned to the kept curves through trial idx.
    idx = meta["idx"].to_numpy()
    target = meta["target"].to_numpy()
    cue = meta["cue"].to_numpy()
    rt = meta["rt"].to_numpy(dtype=float)
    # error rides along like the other labels. trial_type is present only for
    # the value task, so it becomes a coordinate only when the curves carry it.
    error = (meta["error"].to_numpy() if "error" in meta.columns
             else np.full(len(idx), -1))
    has_trial_type = "trial_type" in meta.columns
    if has_trial_type:
        trial_type = meta["trial_type"].to_numpy()
    # treatment is the group label passed by the caller, for example PBS or
    # muscimol. Session provenance rides separately on the session_id coordinate.
    treatment_arr = np.array([treatment] * len(idx), dtype=object).astype(str)

    comp = np.arange(feats["amp"].shape[1])
    coords = dict(
        time=grid,
        comp=comp,
        trial_idx=("trial", idx),
        target=("trial", target),
        error=("trial", error),
        cue=("trial", cue),
        rt=("trial", rt),
        treatment=("trial", treatment_arr),
        session_id=("trial", np.array([session_id] * len(idx))),
    )
    if has_trial_type:
        coords["trial_type"] = ("trial", trial_type)
    ds = xr.Dataset(
        data_vars=dict(
            raw=(("trial", "time"), Y),
            registered=(("trial", "time"), fn),
            warping=(("trial", "time"), gam),
            template=(("time",), template),
            amp_scores=(("trial", "comp"), feats["amp"]),
            phase_scores=(("trial", "comp"), feats["phase"]),
            joint_scores=(("trial", "comp"), feats["joint"]),
        ),
        coords=coords,
        attrs=dict(
            key=key, n_points=int(n_points),
            session_id=session_id, treatment=treatment,
        ),
    )
    return ds


def save_session(
    ds: "xr.Dataset",
    path: Union[str, PathLike],
) -> Union[str, PathLike]:
    """Write a session Dataset to a netCDF file.

    Parameters
    ----------
    ds : xarray.Dataset
        Session dataset to save.
    path : str or path-like
        Output path for the netCDF file.

    Returns
    -------
    str or path-like
        The same path passed to ``path``.
    """
    ds.to_netcdf(path)  # xarray handles the netcdf4 backend
    return path


def load_sessions(
    paths: Sequence[Union[str, PathLike]],
) -> "xr.Dataset":
    """Load session datasets and concatenate them along the trial axis.

    Parameters
    ----------
    paths : sequence of str or path-like
        Paths to netCDF files written by ``save_session``.

    Returns
    -------
    xarray.Dataset
        Pooled dataset with trials concatenated along the ``trial`` dimension.

    Notes
    -----
    The ``treatment`` and ``session_id`` coordinates are retained from each
    session dataset.
    """
    import xarray as xr

    parts = [xr.open_dataset(p) for p in paths]
    pooled = xr.concat(parts, dim="trial")
    return pooled


def mean_curve_by_treatment(
    ds: "xr.Dataset",
    source: str = "registered",
) -> pd.DataFrame:
    """Compute the mean curve for each treatment.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing a ``treatment`` coordinate and the selected curve
        data variable.
    source : str, optional
        Name of the data variable to average. Default is ``"registered"``.

    Returns
    -------
    pandas.DataFrame
        Table indexed by time. Each column contains the mean curve for one
        treatment.

    Raises
    ------
    KeyError
        If ``source`` is not a data variable in ``ds``.
    """
    df = {}
    for tr in np.unique(ds["treatment"].values):
        sel = ds[source].values[ds["treatment"].values == tr]
        df[str(tr)] = sel.mean(axis=0)
    return pd.DataFrame(df, index=ds["time"].values)
