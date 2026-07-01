"""
Shared pytest fixtures for the ERPA unit tests.

The unit tests build small synthetic trials and tables.
"""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def ports3():
    """Three collinear ports 4 cm apart, center evenly between the choices."""
    return {
        "center": np.array([820.0, 360.0]),
        "choice_L": np.array([822.0, 290.0]),   # 70 px below center
        "choice_R": np.array([822.0, 430.0]),   # 70 px above center
    }                                            # L-R = 140 px = 8 cm


def make_trial(idx, choice, ports, n=40, fps=25, dev_px=0.0):
    """
    Build one synthetic trial: a straight slide from center to the chosen port.

    With dev_px set, the path bows toward the unchosen port by that many pixels
    at mid-movement, so deviation measures have a known sign and size.
    """
    center = ports["center"]
    chosen = ports["choice_R"] if choice == 1 else ports["choice_L"]
    unchosen = ports["choice_L"] if choice == 1 else ports["choice_R"]
    ce, ch = 5, n - 5                         # center_exit, choice_entry indices

    xy = np.repeat(center[None, :], n, axis=0).astype(float)
    seg = np.linspace(0, 1, ch - ce + 1)[:, None]
    path = center * (1 - seg) + chosen * seg
    if dev_px:
        # An early excursion toward the unchosen port, then on to the chosen
        # port: a change-of-mind path that yields a positive toward-unchosen
        # deviation, for testing the sign convention.
        u_dir = (unchosen - center) / (np.linalg.norm(unchosen - center) + 1e-9)
        way = center + dev_px * u_dir
        k = max(1, (ch - ce) // 3)
        leg1 = np.linspace(center, way, k + 1)
        leg2 = np.linspace(way, chosen, (ch - ce) - k + 1)
        path = np.vstack([leg1, leg2[1:]])
    xy[ce:ch + 1] = path
    xy[ch + 1:] = chosen

    d = chosen - center
    head = np.full(n, np.arctan2(d[1], d[0]))
    lin = np.zeros(n)
    lin[ce:ch] = 10.0
    return dict(
        idx=idx, centroid=xy, head_angle=head, lin_vel=lin, ang_vel=np.zeros(n),
        nodes=None, t=(np.arange(n) - ce) / fps,
        events=dict(center_entry=1, center_exit=ce, choice_entry=ch,
                    reward_entry=n - 1),
        window=(0, n), framerate=fps,
        target=choice, choice=choice, error=0, cue=4.0,
        rt=0.5, sampling=0.4, session="synthetic",
    )


@pytest.fixture
def synth_trials(ports3):
    """A balanced set of clean synthetic trials."""
    return [make_trial(i, i % 2, ports3) for i in range(12)]


@pytest.fixture
def trial_factory():
    """The make_trial builder, for tests that need custom trials."""
    return make_trial


@pytest.fixture
def behavior_csv(tmp_path):
    """
    Write a small behavioral CSV in the raw column layout, including one
    negative-sampling trial and one long-RT trial that the loader should drop.
    """
    rows = []
    for i in range(20):
        rows.append(dict(sample=0.4, latency=0.5, presented=i % 2,
                         response=i % 2, cue=4, time=10.0 + i,
                         retrieval=0.3, session="s"))
    rows[3]["sample"] = -0.2          # negative sampling, MedPC artifact
    rows[7]["latency"] = 9.0          # long RT, wandering
    df = pd.DataFrame(rows)
    path = tmp_path / "behavior.csv"
    df.to_csv(path)
    return str(path)
