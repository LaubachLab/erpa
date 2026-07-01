"""
Per-trial measure table assembly.

This module builds a trial-level pandas DataFrame from ERPA trial dictionaries.
The table contains identifier and task-label columns, followed by numeric
behavioral, spatial, kinematic, and optional functional-data measures.
"""

import warnings
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from erpa.spatiotemporal.spatial import scalar_feature_matrix
from erpa.util import as_meta_frame


# Identifier and categorical task-label columns. Columns not listed here are
# treated as candidate numeric measures when they have numeric dtype. Behavioral
# scalars such as sampling and RT are measures, not labels. session and subject
# are included so that downstream additions with those names are treated as
# labels; missing columns are ignored.
LABEL_COLUMNS = ("idx", "absolute_trial", "session", "subject",
                 "target", "choice", "trial_type", "error")


_FDA_IMPORT_HINT = (
    "FDA features require optional dependencies. "
    "Install with `pip install erpa[fda]`."
)


def build_measure_table(
    trials: Sequence[Dict[str, Any]],
    ports: Mapping[str, np.ndarray],
    pix: Optional[float] = None,
    node: str = "centroid",
    node_names: Optional[Sequence[str]] = None,
    port_spacing_cm: float = 4.0,
    add_fda: bool = False,
    add_fda_lag: bool = True,
    add_fda_scores: bool = False,
    n_fda_scores: int = 1,
    key: str = "lin_vel",
    n_points: int = 50,
    max_iter: int = 8,
    return_exclusions: bool = False,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Dict[str, List[int]]]]:
    """
    Build a per-trial table of scalar measures.

    The base table is computed by ``scalar_feature_matrix`` and does not require
    optional FDA dependencies. When ``add_fda=True``, movement velocity curves are
    registered with the FDA registration functions, and selected FDA-derived
    measures are merged into the base table by trial ``idx``.

    ``fda_lag`` is the signed area between each trial's warping function and the
    identity line. ``fda_score1``, ``fda_score2``, and later score columns are
    amplitude fPCA scores from the registered velocity curves. These scores are
    relative to the registered dataset used in the current call.

    The scalar feature extractor and the FDA curve extractor use different trial
    inclusion checks. A trial can be retained in the scalar table and omitted from
    FDA registration if its movement segment is not usable for registration. Such
    trials remain in the output table, and their FDA columns are ``NaN``. Set
    ``return_exclusions=True`` to return the trial indices found by only one
    extractor.

    Parameters
    ----------
    trials : list of dict
        Trial dictionaries from ``erpa.core.session.load_session`` or
        ``erpa.core.session.build_trials``.
    ports : dict
        Port locations. Expected keys are ``"center"``, ``"choice_L"``, and
        ``"choice_R"`` when all ports are available.
    pix : float or None, optional
        Pixels-to-centimeters factor. If ``None``, the scale is derived from the
        port layout by the spatial feature functions.
    node : str, optional
        Position source used for path measures. Use ``"centroid"`` for the body
        centroid or a pose-node name for a tracked keypoint.
    node_names : sequence of str or None, optional
        Pose-node names used when ``node`` selects a keypoint from the trial
        node array.
    port_spacing_cm : float, optional
        Physical spacing between adjacent ports in centimeters. Used when
        deriving the pixel-to-centimeter scale from port locations.
    add_fda : bool, optional
        If ``True``, add FDA-derived measures from registered movement curves.
        The optional FDA dependencies must be installed.
    add_fda_lag : bool, optional
        If ``True`` and ``add_fda=True``, add the ``fda_lag`` column.
    add_fda_scores : bool, optional
        If ``True`` and ``add_fda=True``, add amplitude fPCA score columns named
        ``fda_score1``, ``fda_score2``, and so on.
    n_fda_scores : int, optional
        Number of amplitude fPCA score columns to add when
        ``add_fda_scores=True``.
    key : str, optional
        Trial signal used to build movement curves for FDA registration.
    n_points : int, optional
        Number of samples in each resampled movement curve.
    max_iter : int, optional
        Maximum number of elastic-registration iterations passed to
        ``register_all``.
    return_exclusions : bool, optional
        If ``True``, return a second object reporting trial ``idx`` values found
        by only the scalar extractor or only the FDA extractor.

    Returns
    -------
    table : pandas.DataFrame
        Per-trial measure table. Rows are trials retained by the scalar feature
        extractor. Label columns appear first when present, followed by numeric
        measure columns.
    exclusions : dict of list of int
        Returned only when ``return_exclusions=True``. ``"scalar_only"`` contains
        trial ``idx`` values in the scalar table but not in the FDA table.
        ``"fda_only"`` contains trial ``idx`` values in the FDA table but not in
        the scalar table.

    Raises
    ------
    ImportError
        If ``add_fda=True`` and the optional FDA dependencies are not installed.
    ValueError
        If ``add_fda=True`` but both FDA output options are disabled, or if
        ``n_fda_scores < 1``.
    """
    feats = scalar_feature_matrix(trials, ports, node=node,
                                  node_names=node_names, pix=pix,
                                  port_spacing_cm=port_spacing_cm)
    feats["idx"] = feats["idx"].astype(int)
    table = feats
    excl = {"scalar_only": [], "fda_only": []}

    if add_fda:
        if not add_fda_lag and not add_fda_scores:
            raise ValueError(
                "add_fda=True but both add_fda_lag and add_fda_scores are False."
            )
        if n_fda_scores < 1:
            raise ValueError("n_fda_scores must be >= 1.")

        # FDA scalar measures require the optional registration stack. Import
        # here so the base table can be built without those dependencies.
        try:
            from erpa.spatiotemporal.curves import movement_curves
            from erpa.fda.registration import register_all
        except ImportError as exc:
            raise ImportError(_FDA_IMPORT_HINT) from exc

        grid, Y, meta = movement_curves(trials, key=key, n_points=n_points)
        meta = as_meta_frame(meta)
        n_curve = Y.shape[0] if Y.ndim == 2 else 0
        if n_curve >= 3 and n_curve == len(meta):
            # Use at least three components for registration, unless the caller
            # requests more exported fPCA scores.
            n_comp = max(3, n_fda_scores)
            R = register_all(grid, Y, n_comp=n_comp, max_iter=max_iter,
                             phase_joint=False)

            fda_dict = {
                "idx": meta["idx"].astype(int).to_numpy(),
            }

            if add_fda_lag:
                fda_dict["fda_lag"] = np.asarray(R["lag"], float).ravel()

            if add_fda_scores:
                scores = np.asarray(R["amp"], float)
                if scores.ndim == 1:
                    scores = scores[:, None]
                n_scores = min(n_fda_scores, scores.shape[1])
                for j in range(n_scores):
                    fda_dict[f"fda_score{j + 1}"] = scores[:, j]

            # Limit all FDA columns to the shortest returned length before
            # building the DataFrame.
            m = min(len(values) for values in fda_dict.values())
            fda = pd.DataFrame({
                name: values[:m]
                for name, values in fda_dict.items()
            })
            table = table.merge(fda, on="idx", how="left")
            s_ids, f_ids = set(feats["idx"]), set(fda["idx"])
            excl["scalar_only"] = sorted(s_ids - f_ids)
            excl["fda_only"] = sorted(f_ids - s_ids)
            if excl["scalar_only"]:
                warnings.warn(
                    f"{len(excl['scalar_only'])} trials have scalar features but "
                    f"no FDA scalars, dropped by the registration window. Their "
                    f"fda_ columns are NaN, not an error."
                )
        else:
            warnings.warn("FDA scalars skipped: fewer than 3 usable velocity "
                          "curves, or a curve/meta length mismatch.")

    if return_exclusions:
        return table, excl
    return table


def measure_columns(table: pd.DataFrame) -> List[str]:
    """
    Return numeric measure columns from a measure table.

    Identifier and categorical task-label columns listed in ``LABEL_COLUMNS`` are
    excluded. Non-numeric columns are also excluded. Behavioral scalars such as
    sampling and reaction time are retained when present because they are numeric
    measures, not label columns.

    Parameters
    ----------
    table : pandas.DataFrame
        Per-trial measure table from ``build_measure_table``.

    Returns
    -------
    list of str
        Column names for numeric measure variables.
    """
    import pandas as pd
    return [c for c in table.columns
            if c not in LABEL_COLUMNS and pd.api.types.is_numeric_dtype(table[c])]
