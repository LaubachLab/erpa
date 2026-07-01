"""Session loading and the trial pipeline.

The public entry point is load_session, which turns a SLEAP pose file and a
behavioral CSV into analysis-ready trials.
"""

from erpa.core.session import (
    load_session,
    build_trials,
    load_behavior_csv,
    compute_session_series,
    locate_ports,
    ports_array,
    estimate_frame0_time,
    prepare_figure_trials,
)

__all__ = [
    "load_session",
    "build_trials",
    "load_behavior_csv",
    "compute_session_series",
    "locate_ports",
    "ports_array",
    "estimate_frame0_time",
    "prepare_figure_trials",
]
