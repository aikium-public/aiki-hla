"""Tiered-evaluation metrics for AIKI-HLA.

Computes per-allele bootstrap-CI'd AUROC across the four evaluation strata
established in the manuscript Methods §"Cluster-aware splitting and strict
novel-allele stratum":

  broad           — all 263 test alleles (PA-med 0.775 [0.754, 0.806] for the
                    deployed ensemble)
  novel_peptide   — peptide cluster unseen in training (126 alleles)
  novel_allele    — allele cluster unseen in training (57 alleles)
  both_novel      — neither cluster seen in training (128 alleles)
  strict          — strict novel-allele stratum: 9-allele methods-development
                    extension; the headline strict generalization result
                    (PA-med 0.706 [0.648, 0.774])

Bootstrap CIs are computed by resampling alleles within each stratum
(`n_resamples=1000` default; matches the manuscript's reporting).

This is the public-facing API; the five strata above are the
deployment-relevant evaluation surface. Finer-grained diagnostic splits
used during manuscript preparation are not exposed at the package level.

Usage:
    from aiki_hla.evaluation import compute_tiered_metrics
    results = compute_tiered_metrics(predictions_df, gold_df)
    # results = {
    #   "broad":         {"pa_med": 0.775, "ci": [0.754, 0.806], "n_alleles": 263},
    #   "novel_peptide": {...},
    #   "novel_allele":  {...},
    #   "both_novel":    {...},
    #   "strict":        {...},
    # }

  predictions_df  must have columns: peptide, allele, score
  gold_df         must have columns: peptide, allele, binding_label,
                                     eval_bucket (broad/novel_peptide/...)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# Package data
_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_STRICT_STRATUM_PATH = _DATA_DIR / "strict_novel_allele_stratum.json"


def _load_strict_stratum() -> dict:
    """Load the locked 9-allele methods-development-extension allele set.

    The set is locked at the release-corpus version; the JSON file is
    bundled as package data and SHA-pinned in the release manifest.
    """
    return json.loads(_STRICT_STRATUM_PATH.read_text())


def _bootstrap_ci_median(values: list[float], n_resamples: int = 1000, seed: int = 42, alpha: float = 0.05) -> tuple[float, float, float]:
    """Bootstrap a median + 95% CI. Returns (point, ci_low, ci_high)."""
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    medians = np.empty(n_resamples)
    n = len(arr)
    for i in range(n_resamples):
        sample = rng.choice(arr, size=n, replace=True)
        medians[i] = np.median(sample)
    return float(np.median(arr)), float(np.quantile(medians, alpha / 2)), float(np.quantile(medians, 1 - alpha / 2))


def _per_allele_auroc(df: pd.DataFrame, min_per_class: int = 1) -> dict[str, float]:
    """Return {allele: AUROC} for every allele with at least min_per_class of each class."""
    out: dict[str, float] = {}
    for allele, sub in df.groupby("allele"):
        labels = sub["binding_label"].astype(int).values
        n_pos = int(labels.sum())
        n_neg = int(len(labels) - n_pos)
        if n_pos < min_per_class or n_neg < min_per_class:
            continue
        out[allele] = float(roc_auc_score(labels, sub["score"].astype(float).values))
    return out


def compute_tiered_metrics(
    predictions: pd.DataFrame,
    gold: pd.DataFrame,
    strata: Optional[list[str]] = None,
    n_resamples: int = 1000,
    seed: int = 42,
) -> dict:
    """Compute per-stratum bootstrap-CI'd per-allele median AUROC.

    Parameters
    ----------
    predictions
        DataFrame with columns: peptide, allele, score.
    gold
        DataFrame with columns: peptide, allele, binding_label, eval_bucket.
        The eval_bucket column must contain one of:
        'trivial', 'novel_peptide', 'novel_allele', 'both_novel'.
    strata
        Subset of strata to evaluate. Default: all five
        ('broad', 'novel_peptide', 'novel_allele', 'both_novel', 'strict').
    n_resamples
        Bootstrap resamples (default 1000; matches the manuscript).
    seed
        RNG seed for bootstrap reproducibility.

    Returns
    -------
    dict with one entry per stratum:
        {"pa_med": float, "ci_low": float, "ci_high": float, "n_alleles": int}
    """
    if strata is None:
        strata = ["broad", "novel_peptide", "novel_allele", "both_novel", "strict"]

    joined = predictions.merge(gold, on=["peptide", "allele"], how="inner")
    if "binding_label" not in joined.columns or "eval_bucket" not in joined.columns:
        raise ValueError(
            "gold must have columns: peptide, allele, binding_label, eval_bucket"
        )

    results: dict = {}
    strict_alleles = set(_load_strict_stratum()["alleles"]) if "strict" in strata else set()

    for stratum in strata:
        if stratum == "broad":
            sub = joined
        elif stratum in {"novel_peptide", "novel_allele", "both_novel"}:
            sub = joined[joined["eval_bucket"] == stratum]
        elif stratum == "strict":
            sub = joined[joined["allele"].isin(strict_alleles)]
        else:
            raise ValueError(f"unknown stratum: {stratum}")
        per_allele = _per_allele_auroc(sub)
        if not per_allele:
            results[stratum] = {"pa_med": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n_alleles": 0}
            continue
        point, lo, hi = _bootstrap_ci_median(list(per_allele.values()), n_resamples=n_resamples, seed=seed)
        results[stratum] = {
            "pa_med": point,
            "ci_low": lo,
            "ci_high": hi,
            "n_alleles": len(per_allele),
        }
    return results
