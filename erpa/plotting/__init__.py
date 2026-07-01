"""Plotting imports for ERPA.

This subpackage imports functions for trajectory plots, velocity plots, and
functional-data summary plots. Plotting functions require the optional plotting
dependencies.

Notes
-----
Install plotting dependencies with ``pip install erpa[plot]``.

Trajectory plotting functions can use either recorded arena coordinates or a
figure orientation with response ports shown at the top. Figure orientation does
not change analytical coordinates.
"""

from erpa.plotting.trajectories import (
    plot_trajectories,
    plot_velocity_profiles,
    plot_epoch_velocity,
    plot_trial,
    plot_trajectory_grid,
    plot_kinematics_grid,
)
from erpa.plotting.curves import (
    plot_fpca_modes,
    plot_shape_modes,
    plot_event_lock_comparison,
)
from erpa.plotting.corr_clustermap import corr_clustermap

__all__ = [
    "plot_trajectories",
    "plot_velocity_profiles",
    "plot_epoch_velocity",
    "plot_trial",
    "plot_trajectory_grid",
    "plot_kinematics_grid",
    "plot_fpca_modes",
    "plot_shape_modes",
    "plot_event_lock_comparison",
    "corr_clustermap",
]
