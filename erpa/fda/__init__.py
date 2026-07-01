"""Functional data analysis imports.

This subpackage contains optional tools for elastic registration, functional
summaries, variability analysis, and pooled datasets. These tools require the
optional FDA and storage dependencies.

Notes
-----
Importing this subpackage does not import the optional dependencies. Import the
specific module needed for an analysis.

Modules
-------
registration
    Elastic registration functions and supporting utilities.
functional
    Event-locked functional data analysis wrappers.
variability
    Functional principal component analysis and elastic-shape variability
    summaries.
dataset
    Functions for pooling registered sessions into xarray datasets.
"""

try:
    from erpa.fda.registration import (
        register_all,
        convergence_report,
        elastic_register,
        warping_lag,
        shift_lag,
        amplitude_phase_features,
    )
    from erpa.fda.functional import (
        run_fpca,
        event_locked_fda,
        compare_event_locks,
    )
    from erpa.fda.variability import (
        fpca_summary,
        decision_frame_paths,
        elastic_shape_summary,
        cluster_check,
    )
except ImportError:
    pass

try:
    from erpa.fda.dataset import (
        register_to_dataset,
        save_session,
        load_sessions,
        mean_curve_by_treatment,
    )
except ImportError:
    pass
