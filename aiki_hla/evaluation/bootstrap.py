"""Bootstrap confidence-interval helpers.

Used by `tiered_metrics.compute_tiered_metrics` and by
`validation/reproduce_paper_numbers.py`. Kept as a small public surface
so users computing custom metrics can build on the same bootstrap
machinery (matching the manuscript's reporting style).

Public API:
    bootstrap_ci_median(values, n_resamples=1000, alpha=0.05, seed=42)
        -> (point_estimate, ci_low, ci_high)
"""
from __future__ import annotations

import numpy as np


def bootstrap_ci_median(
    values,
    n_resamples: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
):
    """Bootstrap the median of `values` with a (1-alpha) CI.

    Matches the manuscript's reporting style (1000 resamples; alpha=0.05).
    Returns (point, ci_low, ci_high) as Python floats.
    """
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    medians = np.empty(n_resamples)
    for i in range(n_resamples):
        medians[i] = np.median(rng.choice(arr, size=len(arr), replace=True))
    return float(np.median(arr)), float(np.quantile(medians, alpha / 2)), float(np.quantile(medians, 1 - alpha / 2))
