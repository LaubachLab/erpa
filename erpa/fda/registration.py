"""Elastic registration and fPCA features for trial curves.

This module registers resampled trial curves with Fisher-Rao elastic
registration and computes warping lag, amplitude scores, phase scores, and joint
fPCA scores.

Dependencies
------------
numpy
fdasrsf
"""

from typing import Any, Dict, Tuple

import numpy as np
from numpy.typing import ArrayLike
import fdasrsf as fs

_trap = getattr(np, "trapezoid", None) or np.trapz


def register_all(
    grid: ArrayLike,
    Y: ArrayLike,
    n_comp: int = 3,
    max_iter: int = 10,
    phase_joint: bool = True,
) -> Dict[str, np.ndarray]:
    """Register curves and compute amplitude, phase, and joint features.

    Fisher-Rao registration is fit once. The registered amplitude curves,
    warping functions, lag values, and fPCA scores are all derived from that
    fit.

    Parameters
    ----------
    grid : array-like of shape (n_points,)
        Shared time grid for the input curves.
    Y : array-like of shape (n_trials, n_points)
        Resampled trial curves on the shared grid.
    n_comp : int, optional
        Number of fPCA components for amplitude, phase, and joint scores.
    max_iter : int, optional
        Maximum number of iterations passed to ``fdasrsf.fdawarp.srsf_align``.
    phase_joint : bool, optional
        If True, compute phase and joint fPCA scores. If False, return phase
        and joint score arrays filled with NaN values.

    Returns
    -------
    dict
        Registration output and fPCA scores.

        ``fn`` : ndarray of shape (n_trials, n_points)
            Registered amplitude curves.
        ``gam`` : ndarray of shape (n_trials, n_points)
            Warping functions.
        ``lag`` : ndarray of shape (n_trials,)
            Signed area between each warping function and the identity line.
        ``template`` : ndarray of shape (n_points,)
            Karcher mean template returned by ``fdasrsf``.
        ``amp`` : ndarray of shape (n_trials, n_comp)
            Amplitude fPCA scores.
        ``amp_latent`` : ndarray of shape (n_comp,)
            Amplitude fPCA latent values when available; otherwise NaN values.
        ``phase`` : ndarray of shape (n_trials, n_comp)
            Phase fPCA scores. Values are NaN when ``phase_joint`` is False or
            phase fPCA fails.
        ``joint`` : ndarray of shape (n_trials, n_comp)
            Joint fPCA scores. Values are NaN when ``phase_joint`` is False or
            joint fPCA fails.
    """
    f = np.ascontiguousarray(np.asarray(Y, dtype=np.float64).T)
    t = np.ascontiguousarray(np.asarray(grid, dtype=np.float64))
    obj = fs.fdawarp(f, t)
    obj.srsf_align(parallel=False, MaxItr=max_iter, verbose=False)

    fn = obj.fn.T
    gam = obj.gam.T
    lag = _trap(gam - t[None, :], t, axis=1)

    n = gam.shape[0]

    def _nan_scores() -> np.ndarray:
        return np.full((n, n_comp), np.nan, dtype=np.float64)

    # Amplitude PCA supplies amp1, which the scalar table uses.
    vp = fs.fdavpca(obj)
    vp.calc_fpca(no=n_comp)
    amp = np.asarray(vp.coef, dtype=np.float64)

    # Phase and joint PCA require a Karcher mean of the warping functions. That
    # mean can fail to converge for some sessions, and fdasrsf can then raise an
    # IndexError in the SqrtMean loop. The scalar measure table uses lag and amp,
    # so phase and joint scores are optional. lag is computed from gam and is not
    # affected when phase or joint PCA fails.
    phase = _nan_scores()
    joint = _nan_scores()
    if phase_joint:
        try:
            hp = fs.fdahpca(obj)
            hp.calc_fpca(no=n_comp)
            phase = np.asarray(hp.coef, dtype=np.float64)
        except Exception:
            phase = _nan_scores()
        try:
            jp = fs.fdajpca(obj)
            jp.calc_fpca(no=n_comp)
            joint = np.asarray(jp.coef, dtype=np.float64)
        except Exception:
            joint = _nan_scores()

    template = getattr(obj, "mqn", None)
    if template is None:
        template = getattr(obj, "fmean", None)

    amp_latent = np.asarray(getattr(vp, "latent", np.full(n_comp, np.nan)),
                            dtype=np.float64)
    return dict(fn=fn, gam=gam, lag=lag, template=template,
                amp=amp, amp_latent=amp_latent, phase=phase, joint=joint)


def convergence_report(
    grid: ArrayLike,
    Y: ArrayLike,
    iters: Tuple[int, int] = (20, 40),
    n_comp: int = 3,
    tol: float = 0.99,
) -> Dict[str, Any]:
    """Compare registration results across two iteration counts.

    The function fits ``register_all`` at two iteration counts with phase and
    joint fPCA disabled. It compares lag values and the first amplitude fPCA
    score across the two fits.

    Parameters
    ----------
    grid : array-like of shape (n_points,)
        Shared time grid for the input curves.
    Y : array-like of shape (n_trials, n_points)
        Resampled trial curves on the shared grid.
    iters : tuple of int, optional
        Two iteration counts used for the comparison.
    n_comp : int, optional
        Number of amplitude fPCA components computed by ``register_all``.
    tol : float, optional
        Correlation threshold used to mark lag and amp1 as stable.

    Returns
    -------
    dict
        Stability summary.

        ``iters`` : tuple of int
            Iteration counts used for the comparison.
        ``n_trials`` : int
            Number of registered trials.
        ``lag_corr`` : float
            Correlation between lag values from the two fits.
        ``amp1_corr`` : float
            Absolute correlation between first amplitude scores from the two
            fits.
        ``amp1_pc_ratio`` : float
            Ratio of amplitude PC1 latent value to amplitude PC2 latent value
            from the higher-iteration fit.
        ``lag_stable`` : bool
            True when ``lag_corr`` is greater than or equal to ``tol``.
        ``amp1_stable`` : bool
            True when ``amp1_corr`` is greater than or equal to ``tol``.
        ``amp1_axis_well_defined`` : bool
            True when ``amp1_pc_ratio`` is finite and at least 1.5.
    """
    n1, n2 = int(iters[0]), int(iters[1])
    a = register_all(grid, Y, n_comp=n_comp, max_iter=n1, phase_joint=False)
    b = register_all(grid, Y, n_comp=n_comp, max_iter=n2, phase_joint=False)
    lag_corr = float(np.corrcoef(a["lag"], b["lag"])[0, 1])
    amp1_corr = float(abs(np.corrcoef(a["amp"][:, 0], b["amp"][:, 0])[0, 1]))
    lat = np.asarray(b["amp_latent"], float)
    ratio = float(lat[0] / lat[1]) if lat.size >= 2 and lat[1] > 0 else float("nan")
    return {
        "iters": (n1, n2),
        "n_trials": int(np.asarray(a["lag"]).size),
        "lag_corr": lag_corr,
        "amp1_corr": amp1_corr,
        "amp1_pc_ratio": ratio,
        "lag_stable": lag_corr >= tol,
        "amp1_stable": amp1_corr >= tol,
        "amp1_axis_well_defined": bool(np.isfinite(ratio) and ratio >= 1.5),
    }


# ----------------------------------------------------------------------
# Lower-level registration helpers
#
# elastic_register returns the aligned curves, warping functions, and template
# from one fdawarp fit. amplitude_phase_features returns amplitude, phase, and
# joint fPCA scores. warping_lag converts a warping matrix to per-trial timing.
# shift_lag is a shift-only fallback that uses scikit-fda.
# ----------------------------------------------------------------------

def elastic_register(
    grid: ArrayLike,
    Y: np.ndarray,
    max_iter: int = 10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Register curves with Fisher-Rao elastic registration.

    Parameters
    ----------
    grid : array-like of shape (n_points,)
        Shared time grid for the input curves.
    Y : array-like of shape (n_trials, n_points)
        Resampled trial curves on the shared grid.
    max_iter : int, optional
        Maximum number of iterations passed to ``fdasrsf.fdawarp.srsf_align``.

    Returns
    -------
    fn : ndarray of shape (n_trials, n_points)
        Registered amplitude curves.
    gam : ndarray of shape (n_trials, n_points)
        Warping functions.
    template : ndarray of shape (n_points,)
        Karcher mean template returned by ``fdasrsf``.
    """
    f = np.ascontiguousarray(Y.T, dtype=np.float64)  # (M, N)
    t = np.ascontiguousarray(grid, dtype=np.float64)
    obj = fs.fdawarp(f, t)
    obj.srsf_align(parallel=False, MaxItr=max_iter, verbose=False)
    return obj.fn.T, obj.gam.T, obj.mqn if hasattr(obj, "mqn") else obj.fmean


def warping_lag(
    grid: np.ndarray,
    gam: np.ndarray,
) -> np.ndarray:
    """Compute lag from warping functions.

    Lag is the signed area between each warping function and the identity line.
    Positive values indicate a slower curve relative to the template. Negative
    values indicate a faster curve relative to the template.

    Parameters
    ----------
    grid : array-like of shape (n_points,)
        Shared time grid.
    gam : array-like of shape (n_trials, n_points)
        Warping functions.

    Returns
    -------
    ndarray of shape (n_trials,)
        Signed lag values.
    """
    return _trap(gam - grid[None, :], grid, axis=1)


def shift_lag(
    grid: ArrayLike,
    Y: ArrayLike,
) -> np.ndarray:
    """Estimate lag with shift-only registration.

    Parameters
    ----------
    grid : array-like of shape (n_points,)
        Shared time grid for the input curves.
    Y : array-like of shape (n_trials, n_points)
        Resampled trial curves on the shared grid.

    Returns
    -------
    ndarray of shape (n_trials,)
        Per-trial shift estimates from scikit-fda.
    """
    from skfda import FDataGrid
    from skfda.preprocessing.registration import LeastSquaresShiftRegistration
    fd = FDataGrid(data_matrix=Y, grid_points=grid)
    reg = LeastSquaresShiftRegistration()
    reg.fit_transform(fd)
    return np.asarray(reg.deltas_)


def amplitude_phase_features(
    grid: ArrayLike,
    Y: np.ndarray,
    n_comp: int = 3,
    max_iter: int = 8,
) -> Dict[str, np.ndarray]:
    """Compute amplitude, phase, and joint fPCA scores.

    Phase scores describe variation in the time-warping functions. They are not
    Fourier or Hilbert phase values. The warping functions are represented in an
    SRVF tangent space before horizontal fPCA is computed by ``fdasrsf``.

    Parameters
    ----------
    grid : array-like of shape (n_points,)
        Shared time grid for the input curves.
    Y : array-like of shape (n_trials, n_points)
        Resampled trial curves on the shared grid.
    n_comp : int, optional
        Number of fPCA components.
    max_iter : int, optional
        Maximum number of iterations passed to ``fdasrsf.fdawarp.srsf_align``.

    Returns
    -------
    dict
        fPCA score arrays.

        ``amp`` : ndarray of shape (n_trials, n_comp)
            Amplitude fPCA scores.
        ``phase`` : ndarray of shape (n_trials, n_comp)
            Phase fPCA scores.
        ``joint`` : ndarray of shape (n_trials, n_comp)
            Joint fPCA scores.
    """
    f = np.ascontiguousarray(Y.T, dtype=np.float64)
    t = np.ascontiguousarray(grid, dtype=np.float64)
    obj = fs.fdawarp(f, t)
    obj.srsf_align(parallel=False, MaxItr=max_iter, verbose=False)

    vp = fs.fdavpca(obj); vp.calc_fpca(no=n_comp)
    hp = fs.fdahpca(obj); hp.calc_fpca(no=n_comp)
    jp = fs.fdajpca(obj); jp.calc_fpca(no=n_comp)
    return {"amp": vp.coef, "phase": hp.coef, "joint": jp.coef}
