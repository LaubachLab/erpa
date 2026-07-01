"""
erpa.core.config - Rig and pose-model configuration.

Collects the calibration values and node names that are specific to one camera,
one arena, and one SLEAP model into a single object. Adapting ERPA to a new
setup then means building one RigConfig, not hunting down scattered module
constants and default arguments. DEFAULT_CONFIG holds the values for the Laubach
Lab rig used to develop the package. The module-level constants in
erpa.core.session are sourced from DEFAULT_CONFIG, so this object is the single
place those values are defined.

Pass a RigConfig to erpa.core.session.load_session to override the rig for a
session. Explicit keyword arguments to that function still win over the config,
so a one-off change does not require building a new config.

The pixels-to-cm scale is deliberately not a field here. It is derived per
session from the detected port layout and port_spacing_cm, since a fixed factor
is tied to one resolution. See erpa.spatiotemporal.spatial.cm_per_pixel. Note
that load_session derives the pixels-to-cm scale per session from the detected
ports in a two-pass build, so velocities in cm/s use that scale. The
PIXEL_TO_CM_CONVERSION constant in erpa.core.session is only a fallback when
the ports are not both located. Distances from the spatial module always use
the port-derived scale.
"""

from dataclasses import dataclass, replace
from typing import Optional, Tuple


@dataclass(frozen=True)
class RigConfig:
    """
    One session's rig and pose-model settings.

    Build with defaults for the development rig, then override fields for a new
    camera, arena, or SLEAP model, for example
    RigConfig().with_overrides(framerate=30, heading_node=None).
    """

    # Acquisition.
    framerate: int = 25
    reduced_resolution: Tuple[int, int] = (1006, 758)

    # Pose-model node names. heading_node=None selects the ear-perpendicular
    # convention. centroid_nodes=None averages all nodes.
    heading_node: Optional[str] = "MidCann"
    posterior_nodes: Tuple[str, ...] = ("LeftEar", "RightEar")
    centroid_nodes: Optional[Tuple[str, ...]] = None

    # Velocity smoothing, Savitzky-Golay.
    smoothing_window: int = 6
    poly_order: int = 3

    # Pose cleaning.
    pose_outlier_threshold: float = 90.0     # pixel jump that flags a bad frame
    robust_zscore_threshold: float = 4.0     # robust z cutoff for outliers

    # Arena geometry. Adjacent ports are this far apart, used to derive the
    # pixels-to-cm scale per session.
    port_spacing_cm: float = 4.0

    # Behavior validity. Trials above this percentile of sampling or RT are
    # dropped as wandering or invalid.
    valid_pct: float = 95.0

    # Trial window, seconds before center entry and after choice entry.
    pre_s: float = 0.4
    post_s: float = 0.8

    # Velocity-peak detection used by the lab figure functions.
    min_velocity_threshold: float = 9.0      # cm/s floor for a peak
    max_velocity_search_window: int = 30     # frames forward to a peak
    min_velocity_search_window: int = 15     # frames to a velocity minimum
    min_rest_velocity: float = 6.0           # cm/s fallback floor for a rest minimum
    rest_velocity_fraction: float = 0.2      # rest floor as a fraction of the trial peak

    def with_overrides(self, **kw) -> "RigConfig":
        """Return a copy with the named fields replaced."""
        return replace(self, **kw)


DEFAULT_CONFIG = RigConfig()
