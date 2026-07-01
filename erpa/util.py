"""
erpa.util - Small shared helpers
"""

import pandas as pd

# The labels carried for each kept trial. Producers select the subset present.
META_FIELDS = ("idx", "target", "choice", "trial_type", "error",
               "cue", "rt")


def as_meta_frame(meta) -> pd.DataFrame:
    """
    Coerce a meta table to the DataFrame form.

    Accepts a DataFrame, which is returned unchanged, or a list of per-trial
    dicts, which is converted. This replaces the per-module coercion helpers.
    """
    if isinstance(meta, pd.DataFrame):
        return meta
    return pd.DataFrame(list(meta))


def trial_meta(trials, fields=META_FIELDS) -> pd.DataFrame:
    """Build the per-trial meta DataFrame from a list of trial dicts."""
    rows = [{k: t[k] for k in fields if k in t} for t in trials]
    return pd.DataFrame(rows)
