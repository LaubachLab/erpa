# ERPA

[![License: BSD-3](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](LICENSE)
[![tests](https://github.com/LaubachLab/erpa/actions/workflows/tests.yml/badge.svg)](https://github.com/LaubachLab/erpa/actions/workflows/tests.yml)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21109635.svg)](https://doi.org/10.5281/zenodo.21109635)

How should we represent, quantify, and analyze behavior in an operant decision-making task? ERPA addresses this question with tools for pose-based kinematic analysis, functional data analysis, scalar feature extraction, and analysis pipelines for tabular and time-series classification, dimensionality reduction, and sequential sampling models (e.g., DDMs).

ERPA aligns behavioral events with pose-tracking data and extracts movement measures from decision-making sessions. The package was developed for rodent decision-making tasks using SLEAP pose files and behavioral CSV files. The workflow is sufficiently general for other event-aligned pose analyses.

ERPA handles the steps between raw tracking files and analysis-ready data: session loading, timestamp alignment, trial construction, port-location mapping, kinematic and spatial path measures, event-related trajectories, functional data analysis, and plotting. Additional modules provide complete analysis pipelines for PCA ordination, density clustering, tabular classification with SHAP-based feature importance, time-series classification with MultiRocket and shapelets, and permutation-based inference. Drift-diffusion modeling is supported through compatibility with HSSM.

ERPA quantifies the decision trajectory from center exit to choice entry for collinear two-choice port layouts common in rodent studies. Scalar measures capture behavioral timing, spatial path geometry, and movement kinematics. Functional data analysis tools extract event-locked curves, perform elastic registration via the Fisher-Rao metric, and summarize amplitude and phase variation across trials using functional PCA.

The `extras` modules extend the core pipeline into complete analysis workflows. Scalar measures feed into pipelines for PCA ordination, density clustering, tabular classification with SHAP-based feature importance, and permutation-based inference. Movement trajectories feed into time-series classification using MultiRocket and shapelets, both of which provide channel-level interpretability alongside classifier performance.

Our release of the package on July 1, 2026 provides a fully function toolbox for pose analysis from operant decision-making tasks. Step-by-step tutorials covering each stage of the analysis pipeline will be posted shortly.

**NOTE:** The package requires video recordings with **fixed frame rates**. USB cameras that record with variable frame rates are not currently supported.

## Install

Download or clone the ERPA repository from GitHub. Then open a terminal, move into the downloaded repository directory, and install the package.

For example:

```bash
git clone https://github.com/LaubachLab/erpa.git
cd erpa
pip install -e .
```

If you downloaded the repository as a ZIP file instead, unzip it first, then move into the unzipped folder before running:

```bash
pip install -e .
```

The `-e` option installs ERPA in editable mode. This means Python will use the files in this local folder when you import the package.

Optional features can be installed as follows:

```bash
pip install -e ".[fda]"        # scikit-fda, fdasrsf: registration and FDA tools
pip install -e ".[plot]"       # matplotlib, seaborn: plotting functions
pip install -e ".[storage]"    # xarray, netcdf4: pooled registered datasets
pip install -e ".[extras]"     # optional dependencies for example analyses
```

With mamba (or if you use conda, swap conda for mamba below):

```bash
mamba create -n ERPA -c conda-forge python=3.12 numpy pandas scipy h5py
mamba activate ERPA
pip install -e .
```

For notebook use without installing the package, add the repository path before importing ERPA.

```python
import sys
sys.path.insert(0, PATH_TO_REPOSITORY)
import erpa
```

## Getting started

Start with `load_session` in `erpa.core.session`. It loads a SLEAP `.analysis.h5` file and a behavioral CSV, synchronizes the two time bases, and returns an analysis-ready dataset for the rest of the package.

```python
from erpa.core.session import load_session

trials, df, series, ports, frame0 = load_session(
    "session.analysis.h5",
    "behavior.csv",
    frame0_time=5.32,
    trial_offset=0,
)
```

ERPA ships with default settings matched to the Laubach Lab rig. Pass a `RigConfig` from `erpa.core.config` to adapt the framerate, arena dimensions, or pose-model node names to your own setup.

```python
from erpa.core.config import DEFAULT_CONFIG

config = DEFAULT_CONFIG.with_overrides(framerate=30)
trials, df, series, ports, frame0 = load_session(
    "session.analysis.h5",
    "behavior.csv",
    frame0_time=5.32,
    config=config,
)
```

`frame0_time` is the video timestamp, in seconds, corresponding to the first behavioral trial. It must be measured from the video directly and passed to `load_session`. ERPA includes an experimental helper, `estimate_frame0_time`, but any value it produces should be verified against the video before use in final analyses.

## Measure tables

`build_measure_table` returns one row per trial as a pandas DataFrame. It includes animal and session identifiers along with behavioral, spatial, kinematic, and functional data measures. `measure_columns(table)` returns the numeric analysis columns and excludes identifiers and categorical labels, making it easier to pass the table to PCA, clustering, classification, or modeling code without accidentally using target labels as predictors.

```python
from erpa.spatiotemporal.measures import build_measure_table, measure_columns

table = build_measure_table(trials, ports)
table_fda = build_measure_table(trials, ports, add_fda=True)
X = table[measure_columns(table)]
```

The base measure table uses NumPy, pandas, SciPy, and h5py. `add_fda=True` requires the FDA dependencies and raises an `ImportError` if they are not installed.

## Plotting spatial trajectories

Trajectory plots show the animal's 2D movements for each trial in your dataset.

Note: The default `"paper"` orientation assumes the ports are located on the right side of the raw video recording, which matches the recording setup used in our lab. If your rig uses a different camera orientation, use `orientation="raw"` or adjust the plotting transform before using paper-style trajectory figures.

```python
from erpa.core.session import prepare_figure_trials
from erpa.plotting.trajectories import plot_trajectories

fig = plot_trajectories(
    prepare_figure_trials(trials),
    orientation="paper",
)
```

## FDA workflow

FDA tools are useful for analyzing the shape and timing of movement curves. The fdasrsf package is used for these analyses: https://github.com/jdtuck/fdasrsf_python

A typical FDA workflow is:

1. extract event-locked movement curves;
2. register curves with Fisher-Rao elastic registration;
3. summarize amplitude and phase variation;
4. add selected FDA summaries to the trial-level measure table;
5. optionally pool registered sessions into an xarray Dataset.

The FDA outputs can be used directly for functional summaries or added to downstream models and classifiers.

ERPA uses Fisher-Rao/SRVF elastic registration to separate amplitude variation (curve shape after alignment) from phase variation (the warping needed to align curves). For background on elastic registration, see Srivastava et al. (2011) and Tucker et al. (2013). For background on functional data analysis, see Marron et al. (2015).

References:

Srivastava, A., Wu, W., Kurtek, S., Klassen, E., & Marron, J. S. (2011). Registration of functional data using Fisher-Rao metric. arXiv preprint arXiv:1103.3817.

Tucker, J. D., Wu, W., & Srivastava, A. (2013). Generative models for functional data using phase and amplitude separation. Computational Statistics & Data Analysis, 61, 50-66.

Marron, J. S., Ramsay, J. O., Sangalli, L. M., & Srivastava, A. (2015). Functional data analysis of amplitude and phase variation. Statistical Science, 468-484.

## Modeling and classification workflows

The `extras/` directory includes analyses built on top of the ERPA API:

- tabular classification using geometric and kinematic features
- time-series classification using continuous measures such as linear and angular velocity
- PCA and varimax ordination of geometric and kinematic features
- cluster analysis of geometric and kinematic features
- cluster analysis of registered FDA curves
- partial correlation analysis with permutation testing (e.g., to examine correlation between the properties of a stimulus and a geometric or kinematic measure with the effect of RT factored out)

## Layout

```text
erpa/
  core/
    config.py            RigConfig and pose settings
    session.py           session loading, synchronization, and trial construction

  spatiotemporal/
    spatial.py           geometric and kinematic trial measures
    curves.py            event-locked movement curves and keypoint speed curves
    measures.py          per-trial measure table assembly

  fda/
    registration.py      elastic registration and registration summaries
    functional.py        event-locked FDA wrappers
    variability.py       fPCA and elastic-shape variability summaries
    dataset.py           registered-session pooling into xarray Datasets

  plotting/
    trajectories.py      trajectory, landmark, and velocity plots
    curves.py            registered-curve and functional-data plots

  extras/
    classification/
      classification.py   gradient boosting with SHAP-based feature importance

    clustering/
      clustering.py        PaCMAP and HDBSCAN clustering of measure tables
      fda_clustering.py    K-medoids clustering of registered curves and FDA scores

    dimensionality_reduction/
      pca_ordination.py    PCA and varimax ordination of measure tables

    stats/
      permutation.py       rank effect sizes and condition-permutation tests

    tsc/
      tsc.py               time-series classification using MultiRocket and Shapelets

tests/
  ...
```

## Testing

Run the test suite from the repository root.

```bash
pytest
```

The tests are intended to check the main numerical and data-assembly routines. They are not a substitute for session-level quality control. For real datasets, always inspect synchronization, port detection, trajectory orientation, trial counts, and the relationship between behavioral and video-derived timing.

## Authors

ERPA began as a single Python module developed by Jason Blackmer for data structures, plotting routines, and velocity-curve analysis. Mark Laubach later expanded the module into a package for event-related pose analysis, geometric feature extraction, functional data analysis, and outputs suitable for tabular and time-series classifiers. Kevin Chavez Lopez and Sarah Katz provided feedback and tested early versions during package development. Video data used to develop the package was collected by Kevin Chavez Lopez. The research team is based in the Department of Neuroscience at American University.

## Development notes

Claude (Anthropic), ChatGPT (OpenAI), and Gemini (Google) were used as coding and documentation assistants during development of this package. AI assistance included code refactoring, documentation revision, test-design suggestions, and algorithmic review. All code and documentation were reviewed, edited, and approved by the human authors, who take responsibility for the final package.
