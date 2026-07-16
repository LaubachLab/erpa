"""Unit tests"""

import warnings

import numpy as np
import pandas as pd
import pytest

from erpa.core.config import RigConfig, DEFAULT_CONFIG
from erpa.core.session import (load_behavior_csv, locate_ports,
                               prepare_figure_trials)
from erpa.spatiotemporal.spatial import (
    cm_per_pixel, decision_axis, scalar_feature_matrix, deviation_curves)
from erpa.spatiotemporal.measures import build_measure_table, measure_columns


# --- config ----------------------------------------------------------------

def test_config_defaults():
    assert DEFAULT_CONFIG.framerate == 25
    assert DEFAULT_CONFIG.port_spacing_cm == 4.0
    assert DEFAULT_CONFIG.heading_node == "MidCann"


def test_config_is_frozen():
    with pytest.raises(Exception):
        DEFAULT_CONFIG.framerate = 30


def test_config_with_overrides():
    cfg = DEFAULT_CONFIG.with_overrides(framerate=30, heading_node=None)
    assert cfg.framerate == 30
    assert cfg.heading_node is None
    assert DEFAULT_CONFIG.framerate == 25       # original untouched


def test_session_constants_track_config():
    from erpa.core import session
    assert session.DEFAULT_FRAMERATE == DEFAULT_CONFIG.framerate
    assert session.POSE_OUTLIER_THRESHOLD == DEFAULT_CONFIG.pose_outlier_threshold


# --- behavior loader --------------------------------------------------------

def test_load_behavior_filters_and_anchor(behavior_csv):
    df = load_behavior_csv(behavior_csv)
    # negative-sampling and long-RT trials dropped
    assert (df["sampling"] >= 0).all()
    assert len(df) < 20
    # renamed and derived columns present
    for col in ("sampling", "rt", "target", "choice", "error"):
        assert col in df.columns
    # anchor pinned to the first trial before filtering, time 10.0
    assert df.attrs["t0_anchor"] == pytest.approx(10.0)


def test_load_behavior_trial_offset_moves_anchor(behavior_csv):
    df = load_behavior_csv(behavior_csv, trial_offset=2)
    assert df.attrs["t0_anchor"] == pytest.approx(12.0)


# --- spatial ---------------------------------------------------------------

def test_cm_per_pixel(ports3):
    cpp = cm_per_pixel(ports3)              # 8 cm over 140 px
    assert cpp == pytest.approx(8.0 / 140.0, rel=1e-6)
    c, u, dR, dL = decision_axis(ports3)
    assert abs(dR) * cpp == pytest.approx(4.0, abs=0.1)
    assert abs(dL) * cpp == pytest.approx(4.0, abs=0.1)


def test_cm_per_pixel_rejects_coincident_ports(ports3):
    bad = dict(ports3, choice_R=ports3["choice_L"].copy())
    with pytest.raises(ValueError):
        cm_per_pixel(bad)


def test_scalar_feature_matrix_straight_path(synth_trials, ports3):
    feats = scalar_feature_matrix(synth_trials, ports3)   # pix auto-derived
    assert len(feats) == len(synth_trials)
    for col in ("dev_unchosen_cm", "path_len_cm", "tortuosity", "head_to_chosen"):
        assert col in feats.columns
    # straight slide to the chosen port: little excursion, near-straight, heading
    # pointed at the chosen port
    assert feats["dev_unchosen_cm"].abs().median() < 0.5
    assert feats["tortuosity"].median() == pytest.approx(1.0, abs=0.05)
    assert feats["head_to_chosen"].median() > 0.9
    assert feats["path_len_cm"].median() == pytest.approx(4.0, abs=0.6)


def test_scalar_feature_matrix_signed_deviation(ports3, trial_factory):
    # a path that bows toward the unchosen port should read positive deviation
    trials = [trial_factory(i, i % 2, ports3, dev_px=20.0) for i in range(6)]
    feats = scalar_feature_matrix(trials, ports3)
    assert feats["dev_unchosen_cm"].mean() > 0


def test_deviation_curves_shape(synth_trials, ports3):
    grid, Y, meta = deviation_curves(synth_trials, ports3, n_points=30)
    assert grid.shape == (30,)
    assert Y.shape[1] == 30
    assert len(meta) == Y.shape[0]


# --- measure table ----------------------------------------------------------

def test_build_measure_table_no_fda(synth_trials, ports3):
    tab = build_measure_table(synth_trials, ports3, add_fda=False)
    assert len(tab) == len(synth_trials)
    assert "idx" in tab.columns
    assert set(tab["idx"]) == {t["idx"] for t in synth_trials}
    assert "fda_lag" not in tab.columns


def test_build_measure_table_exclusions_empty(synth_trials, ports3):
    tab, excl = build_measure_table(synth_trials, ports3, add_fda=False,
                                    return_exclusions=True)
    assert excl["scalar_only"] == []
    assert excl["fda_only"] == []


def test_measure_columns_excludes_labels(synth_trials, ports3):
    tab = build_measure_table(synth_trials, ports3, add_fda=False)
    cols = measure_columns(tab)
    assert "idx" not in cols
    assert "dev_unchosen_cm" in cols


def test_measure_columns_keeps_behavioral_scalars(synth_trials, ports3):
    # sampling and RT are behavioral scalar measures, not labels, so they must
    # appear both in the table and among the numeric measure columns, alongside
    # the video-derived timing t_choice
    tab = build_measure_table(synth_trials, ports3, add_fda=False)
    cols = measure_columns(tab)
    for col in ("sampling", "rt", "t_choice"):
        assert col in tab.columns
        assert col in cols
    # categorical and identifier labels stay out of the measure columns
    for col in ("idx", "target", "choice", "error"):
        assert col not in cols


# --- figure-field bridge ----------------------------------------------------

def test_prepare_figure_trials_adds_keys(synth_trials):
    leg = prepare_figure_trials(synth_trials)
    assert len(leg) == len(synth_trials)
    for k in ("trajectory", "velocity", "error", "lmax_idx", "localmin"):
        assert k in leg[0]
    assert leg[0]["error"] == synth_trials[0]["error"]
    # the inputs are not mutated
    assert "trajectory" not in synth_trials[0]


def test_as_meta_frame_coerces():
    from erpa.util import as_meta_frame
    lst = [{"idx": 0, "hit": 1}, {"idx": 1, "hit": 0}]
    df = as_meta_frame(lst)
    assert isinstance(df, pd.DataFrame)
    assert list(df["idx"]) == [0, 1]
    assert as_meta_frame(df) is df          # a DataFrame passes through unchanged


# --- port location ----------------------------------------------------------

def _mislabel(trial, target):
    """Set a trial's target independently of its choice, and recode error.

    The synthetic trial builder walks the path to ``choice``. Overriding
    ``target`` afterward gives an error trial whose geometry is right and whose
    metadata disagrees with the entered side.
    """
    trial["target"] = target
    trial["error"] = int(trial["choice"] != target)
    return trial


def _spread_trials(trial_factory, ports, spec, spread_px=30.0):
    """Build trials whose choice-entry points scatter along the decision axis.

    ``spec`` is a sequence of ``(choice, target)`` pairs. Within each entered
    side the endpoints are offset by a symmetric ladder, so the median endpoint
    of that side lands exactly on the true port. Real choice-entry positions
    scatter this way. The scatter is what lets a contaminated bin pull its
    median off the port, so a test without it cannot see the bug.
    """
    trials = []
    for side in (0, 1):
        rows = [(i, c, tg) for i, (c, tg) in enumerate(spec) if c == side]
        if not rows:
            continue
        offs = (np.linspace(-spread_px, spread_px, len(rows)) if len(rows) > 1
                else np.zeros(1))
        for (i, c, tg), dy in zip(rows, offs):
            shifted = {k: (v + np.array([0.0, dy]) if k.startswith("choice")
                           else v)
                       for k, v in ports.items()}
            trials.append(_mislabel(trial_factory(i, c, shifted), target=tg))
    return sorted(trials, key=lambda t: t["idx"])


def test_locate_ports_bins_by_choice(trial_factory, ports3):
    """Ports follow the entered port, not the correct port.

    Every trial here is a right-target trial and half are errors to the left.
    Binning by target puts all twelve paths in the right bin, leaves the left
    bin empty, and drags the right estimate toward the midpoint of the two
    ports. Binning by choice recovers both ports exactly.
    """
    trials = [_mislabel(trial_factory(i, i % 2, ports3), target=1)
              for i in range(12)]
    assert sum(t["error"] for t in trials) == 6

    ports = locate_ports(trials)

    assert set(ports) == {"center", "choice_L", "choice_R"}
    assert ports["choice_L"] == pytest.approx(ports3["choice_L"])
    assert ports["choice_R"] == pytest.approx(ports3["choice_R"])
    assert ports["center"] == pytest.approx(ports3["center"])


def test_locate_ports_asymmetric_errors_keep_geometry(trial_factory, ports3):
    """Frequent one-sided errors do not bias the port-derived geometry.

    Twelve right-target trials include four errors. Eight left-target trials
    include one. That asymmetry is what collapses one center-to-port distance
    when trials are binned by target. The pixel scale and both distances must
    survive it.
    """
    spec = ([(1, 1)] * 8 + [(0, 1)] * 4          # right target, 4 errors
            + [(0, 0)] * 7 + [(1, 0)] * 1)       # left target, 1 error
    trials = _spread_trials(trial_factory, ports3, spec)
    assert sum(t["error"] for t in trials) == 5

    ports = locate_ports(trials)
    assert ports["choice_L"] == pytest.approx(ports3["choice_L"], abs=1e-6)
    assert ports["choice_R"] == pytest.approx(ports3["choice_R"], abs=1e-6)

    assert cm_per_pixel(ports) == pytest.approx(cm_per_pixel(ports3), rel=1e-9)
    _c, _u, dR, dL = decision_axis(ports)
    assert abs(dR) == pytest.approx(abs(dL), rel=1e-9)      # 70 px each way


def test_locate_ports_omits_a_side_with_no_choices(trial_factory, ports3):
    """A side the rat never entered yields no port, not a wrong one."""
    trials = [_mislabel(trial_factory(i, 1, ports3), target=i % 2)
              for i in range(6)]
    ports = locate_ports(trials)
    assert "choice_L" not in ports
    assert ports["choice_R"] == pytest.approx(ports3["choice_R"])
