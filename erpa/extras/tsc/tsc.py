"""
Time-series classification for ERPA trials.

This module builds multichannel time-series arrays for aeon-compatible
classifiers and provides two feature-extraction pipelines:

MultiRocket transform
    Convolves each channel with a large bank of random kernels and extracts
    summary statistics. Produces a high-dimensional feature matrix that
    feeds a downstream classifier. Fast and consistently accurate.

RandomShapelet transform
    Finds short subsequences (shapelets) that best discriminate classes.
    Produces a low-dimensional feature matrix whose columns have direct
    interpretability as pattern matches in named time-series channels.

Both pipelines produce a flat feature matrix compatible with
``erpa.extras.classification.classification.classifier_importance`` and
``cross_val_score``, so SHAP and permutation analysis apply to either.
"""

import numpy as np
import pandas as pd

from erpa.spatiotemporal.curves import movement_curves
from erpa.spatiotemporal.spatial import deviation_curves, arclength_signal
from erpa.util import as_meta_frame


DEFAULT_CHANNELS = ("time:lin_vel", "time:ang_vel",
                    "space:deviation", "space:speed")


def _channel(trials, ports, name, n_points, node, node_names):
    """
    Build one named time-series channel.

    Parameters
    ----------
    trials : list of dict
        ERPA trial dictionaries.
    ports : dict
        Port locations used by spatial-frame channels.
    name : str
        Channel name. Supported values are ``"time:lin_vel"``,
        ``"time:ang_vel"``, ``"space:deviation"``, and ``"space:speed"``.
    n_points : int
        Number of samples in the resampled channel.
    node : str
        Position source used by spatial-frame channels.
    node_names : sequence of str or None
        Pose-node names used when ``node`` selects a keypoint.

    Returns
    -------
    grid : numpy.ndarray
        Channel grid.
    Y : numpy.ndarray
        Channel matrix with one row per kept trial.
    meta : pandas.DataFrame
        Trial metadata for the kept channel rows.

    Raises
    ------
    ValueError
        If ``name`` is not a supported channel.
    """
    if name == "time:lin_vel":
        return movement_curves(trials, key="lin_vel", n_points=n_points)
    if name == "time:ang_vel":
        return movement_curves(trials, key="ang_vel", n_points=n_points)
    if name == "space:deviation":
        return deviation_curves(trials, ports, n_points=n_points, node=node,
                                node_names=node_names)
    if name == "space:speed":
        return arclength_signal(trials, key="lin_vel", n_points=n_points,
                                node=node, node_names=node_names)
    raise ValueError(f"unknown channel '{name}'")


def assemble_channels(trials, ports, node_names=None, n_points=60,
                      channels=DEFAULT_CHANNELS, node="centroid"):
    """
    Assemble multiple channels into a classifier-ready array.

    Each channel is built separately and indexed by trial ``idx``. The returned
    array keeps only trials present in every requested channel.

    Parameters
    ----------
    trials : list of dict
        ERPA trial dictionaries.
    ports : dict
        Port locations used by spatial-frame channels.
    node_names : sequence of str or None, optional
        Pose-node names used when ``node`` selects a keypoint.
    n_points : int, optional
        Number of samples in each channel.
    channels : sequence of str, optional
        Channel names to build and stack.
    node : str, optional
        Position source used by spatial-frame channels.

    Returns
    -------
    X : numpy.ndarray
        Multichannel array with shape ``(n_kept, n_channels, n_points)`` and
        dtype ``float32``.
    channel_names : list of str
        Channel names in the same order as the second axis of ``X``.
    meta : pandas.DataFrame
        Metadata for the kept trials, in the same row order as ``X``.

    Notes
    -----
    Time-frame channels normalize the movement segment duration. Space-frame
    channels normalize path length. Because each channel has its own trial
    inclusion checks, the final array uses the intersection of trial indices.
    """
    built, metas = {}, {}
    for name in channels:
        _, Y, m = _channel(trials, ports, name, n_points, node, node_names)
        m = as_meta_frame(m)
        built[name] = pd.DataFrame(Y, index=m["idx"].to_numpy())
        metas[name] = m.set_index("idx")

    common = None
    for name in channels:
        ids = set(built[name].index)
        common = ids if common is None else (common & ids)
    common = sorted(common)

    arrs = [built[name].loc[common].values for name in channels]
    X = np.stack(arrs, axis=1).astype(np.float32)
    meta = metas[channels[0]].loc[common].reset_index()
    return X, list(channels), meta


def multirocket_transform(X, n_kernels=2000, random_state=0):
    """
    Apply the MultiRocket transform to a multichannel time-series array.

    MultiRocket convolves each channel with a large bank of random kernels and
    extracts two summary statistics per kernel (PPV and max), producing a
    flat feature matrix suitable for a downstream linear or gradient-boosting
    classifier.

    Parameters
    ----------
    X : numpy.ndarray
        Multichannel array with shape ``(n_cases, n_channels, n_points)`` and
        dtype ``float32``, as returned by ``assemble_channels``.
    n_kernels : int, optional
        Number of random convolutional kernels per channel.
    random_state : int, optional
        Random seed for kernel generation.

    Returns
    -------
    X_features : numpy.ndarray
        Feature matrix with shape ``(n_cases, n_features)`` and dtype
        ``float32``.

    Notes
    -----
    The feature count is approximately
    ``n_kernels * n_channels * n_features_per_kernel``. MultiRocket is
    imported from ``aeon`` inside this function.
    """
    from aeon.transformations.collection.convolution_based import MultiRocket

    mr = MultiRocket(n_kernels=n_kernels, random_state=random_state)
    mr.fit(X)
    return mr.transform(X), mr


def shapelet_transform(X, y, channel_names, n_shapelet_samples=500,
                       max_shapelets=20, random_state=0, n_jobs=1):
    """
    Apply the RandomShapelet transform to a multichannel time-series array.

    Shapelets are short subsequences that best discriminate classes by
    information gain. The transform returns one feature per shapelet, equal to
    the minimum z-normalised Euclidean distance from that shapelet to each
    trial. The fitted shapelets are returned with their channel name, position,
    length, and quality so the most discriminative patterns can be visualised.

    Parameters
    ----------
    X : numpy.ndarray
        Multichannel array with shape ``(n_cases, n_channels, n_points)`` and
        dtype ``float32``, as returned by ``assemble_channels``. Cast to
        ``float64`` internally before fitting to satisfy aeon's Numba backend.
    y : array-like
        Class labels aligned to rows of ``X``.
    channel_names : list of str
        Channel names aligned to the channel axis of ``X``, used to annotate
        the returned shapelet summary.
    n_shapelet_samples : int, optional
        Number of candidate shapelets evaluated during fitting.
    max_shapelets : int, optional
        Maximum number of shapelets retained.
    random_state : int, optional
        Random seed for shapelet sampling.
    n_jobs : int, optional
        Number of parallel jobs for shapelet fitting.

    Returns
    -------
    X_features : numpy.ndarray
        Feature matrix with shape ``(n_cases, n_shapelets_found)``.
    shapelet_summary : pandas.DataFrame
        One row per fitted shapelet with columns ``quality``, ``length``,
        ``position``, ``channel``, and ``series``, where ``series`` is the
        raw shapelet array. Sorted by quality descending.
    transformer : RandomShapeletTransform
        The fitted transformer, for transforming new data.

    Notes
    -----
    ``RandomShapeletTransform`` is imported from ``aeon`` inside this function.
    """
    from aeon.transformations.collection.shapelet_based import (
        RandomShapeletTransform,
    )

    X64 = X.astype(np.float64)
    rst = RandomShapeletTransform(
        n_shapelet_samples=n_shapelet_samples,
        max_shapelets=max_shapelets,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    rst.fit(X64, y)
    X_features = rst.transform(X64)

    rows = []
    for s in rst.shapelets:
        quality, length, position, channel = s[0], s[1], s[2], s[3]
        rows.append(dict(
            quality=float(quality),
            length=int(length),
            position=int(position),
            channel=channel_names[int(channel)],
            series=s[6] if len(s) > 6 else np.array(s[-1]),
        ))
    summary = (pd.DataFrame(rows)
               .sort_values("quality", ascending=False)
               .reset_index(drop=True))
    return X_features, summary, rst
