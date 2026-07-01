# ERPA examples

Project analyses that build on the ERPA package API but are not part of the core
installable package. They are kept here for reference and reproducibility.

- `classification/` tabular classification of the measure table
- `clustering/` density clustering of the measure space and clustering FDA curves
- `dimensionality_reduction/` PCA/varimax ordination of the measure table
- `tsc/` multichannel assembly and time-series classification
- `stats/` rank statistics and condition permutation tests

These import from the public API, for example `from erpa.spatiotemporal.measures
import build_measure_table`. Install the extras they need with
`pip install -e ".[examples]"`.
