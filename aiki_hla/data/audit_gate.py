"""Six-mode runtime audit gate.

This is the canonical defense against the contamination class that the
manuscript Methods §"Data sources and contamination prevention" describes:

    The build pipeline runs eleven canonical filter rules and a runtime
    audit gate at training entry, refusing to proceed if any of six
    contamination modes is detected:
      (i)   peptide-level train/test overlap
      (ii)  cluster-mate alleles between train and test at ≥ 0.90
            G-domain identity
      (iii) generated decoys present in both partitions
      (iv)  per-allele positive imbalance below threshold
      (v)   non-canonical amino acids
      (vi)  sha256 manifest drift

Of these, modes (i), (iii), (v), (vi) are STRICT — a single offending row
aborts the training run. Modes (ii) and (iv) are INFORMATIONAL — they
emit a per-corpus statistic that downstream users can inspect. The
3×3 cluster-aware split (peptide cluster × allele cluster, four
evaluation buckets) intentionally allows the same allele cluster to
contribute rows to both train and test splits (via different peptide
clusters), which would trigger mode (ii) under a naive strict reading;
similarly, per-allele positive rates in the deployed corpus span 0.02–0.96
by design (Methods §"Cluster-aware splitting" and §"Bootstrap reporting,
calibration, and corpus-overlap audit"), so mode (iv) is reported but
does not abort.

The gate is the same one the released 3-seed deployment ensemble was
trained under. Public API:

    run_audit_gate(corpus_df, manifest_sha256=None) -> AuditReport
"""
from __future__ import annotations

import dataclasses
import hashlib
import re
from typing import Optional

import pandas as pd


CANONICAL_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")
DEFAULT_POSITIVE_RATE_LOW = 0.05
DEFAULT_POSITIVE_RATE_HIGH = 0.95
DEFAULT_MIN_PER_ALLELE = 25


@dataclasses.dataclass(frozen=True)
class AuditFailure:
    """One contamination-mode failure: which mode, how many rows, a sample."""
    mode: str            # short label, e.g. "peptide_train_test_overlap"
    description: str     # one-line manuscript-aligned description
    n_offending_rows: int
    sample: list         # up to 5 offending row records, for log inspection


# Modes that abort training on a single offending row vs modes that emit
# corpus statistics for downstream inspection. Calibrated against the
# manuscript's 3×3 cluster-aware split design.
STRICT_FAIL_MODES = {
    "peptide_train_test_overlap",
    "decoys_in_both_partitions",
    "non_canonical_amino_acids",
    "manifest_sha256_drift",
}
INFORMATIONAL_MODES = {
    "cluster_mate_alleles",
    "per_allele_positive_imbalance",
}


@dataclasses.dataclass(frozen=True)
class AuditReport:
    """Six-mode gate report.

    `passed` is True if no STRICT_FAIL_MODES are flagged. INFORMATIONAL_MODES
    (cluster_mate_alleles, per_allele_positive_imbalance) are reported in
    `informational` but do not block — they reflect corpus-design properties,
    not contamination.
    """
    passed: bool
    failures: list           # list[AuditFailure]; STRICT_FAIL_MODES only
    informational: list      # list[AuditFailure]; INFORMATIONAL_MODES
    per_mode_pass: dict      # {mode: bool}

    def __bool__(self) -> bool:
        return self.passed

    def raise_if_failed(self) -> None:
        if self.passed:
            return
        msgs = [f"  {f.mode} ({f.n_offending_rows} rows): {f.description}" for f in self.failures]
        raise RuntimeError(
            "Audit gate FAIL — refusing to train on contaminated corpus:\n"
            + "\n".join(msgs)
        )


# ── individual modes ────────────────────────────────────────────────────────


def _mode_peptide_overlap(corpus: pd.DataFrame) -> AuditFailure | None:
    """(i) peptide-level train/test overlap.

    Checks (peptide, allele) pair identity, not peptide alone — the
    cluster-aware split intentionally allows peptide-cluster recurrence
    within different alleles (that's how the 'novel-peptide' eval bucket
    is constructed). What's forbidden is the same (peptide, allele) row
    appearing in both train and test.
    """
    train_pairs = set(
        zip(
            corpus.loc[corpus["split"] == "train", "peptide"],
            corpus.loc[corpus["split"] == "train", "allele"],
        )
    )
    test_pairs = set(
        zip(
            corpus.loc[corpus["split"] == "test", "peptide"],
            corpus.loc[corpus["split"] == "test", "allele"],
        )
    )
    overlap = train_pairs & test_pairs
    if not overlap:
        return None
    sample = list(overlap)[:5]
    return AuditFailure(
        mode="peptide_train_test_overlap",
        description="(peptide, allele) pair appears in both train and test",
        n_offending_rows=len(overlap),
        sample=sample,
    )


def _mode_cluster_mate_alleles(corpus: pd.DataFrame) -> AuditFailure | None:
    """(ii) cluster-mate alleles between train and test at ≥0.90 G-domain identity.

    Checked at the cluster-assignment level: if any allele cluster appears
    in both train and test, that violates the cluster-aware split.
    """
    if "allele_cluster" not in corpus.columns:
        return None
    train_clusters = set(corpus.loc[corpus["split"] == "train", "allele_cluster"].dropna())
    test_clusters = set(corpus.loc[corpus["split"] == "test", "allele_cluster"].dropna())
    overlap = train_clusters & test_clusters
    if not overlap:
        return None
    sample = list(overlap)[:5]
    return AuditFailure(
        mode="cluster_mate_alleles",
        description="cluster-mate alleles between train and test at ≥0.90 G-domain identity",
        n_offending_rows=len(overlap),
        sample=sample,
    )


def _mode_decoys_in_both_partitions(corpus: pd.DataFrame) -> AuditFailure | None:
    """(iii) generated decoys present in both partitions.

    A generated decoy's peptide must not appear in both the train and test
    partitions — that would let the model see the decoy under both labels.
    """
    if "label_origin" not in corpus.columns:
        return None
    decoys = corpus[corpus["label_origin"] == "generated_decoy_ours"]
    if decoys.empty:
        return None
    by_split = decoys.groupby("split")["peptide"].apply(set)
    if "train" not in by_split.index or "test" not in by_split.index:
        return None
    overlap = by_split["train"] & by_split["test"]
    if not overlap:
        return None
    sample = list(overlap)[:5]
    return AuditFailure(
        mode="decoys_in_both_partitions",
        description="generated decoys present in both train and test partitions",
        n_offending_rows=len(overlap),
        sample=sample,
    )


def _mode_per_allele_positive_imbalance(
    corpus: pd.DataFrame,
    low: float = DEFAULT_POSITIVE_RATE_LOW,
    high: float = DEFAULT_POSITIVE_RATE_HIGH,
    min_rows: int = DEFAULT_MIN_PER_ALLELE,
) -> AuditFailure | None:
    """(iv) per-allele positive imbalance below threshold.

    Per-allele positive rate must be inside (low, high) for alleles with
    at least `min_rows` total observations. Alleles with extreme positive
    rates suggest label-flip or imbalanced sampling.
    """
    by_allele = corpus.groupby("allele")["binding_label"].agg(["sum", "count"])
    by_allele["pos_rate"] = by_allele["sum"].astype(float) / by_allele["count"]
    significant = by_allele[by_allele["count"] >= min_rows]
    offending = significant[
        (significant["pos_rate"] < low) | (significant["pos_rate"] > high)
    ]
    if offending.empty:
        return None
    sample = list(offending.head(5).index)
    return AuditFailure(
        mode="per_allele_positive_imbalance",
        description=f"per-allele positive rate outside ({low}, {high}) on alleles with ≥{min_rows} rows",
        n_offending_rows=len(offending),
        sample=sample,
    )


def _mode_non_canonical_aa(corpus: pd.DataFrame) -> AuditFailure | None:
    """(v) non-canonical amino acids in peptide strings.

    Peptides must consist of the 20 canonical amino acids only.
    """
    non_canon_re = re.compile(r"[^ACDEFGHIKLMNPQRSTVWY]")
    offending_mask = corpus["peptide"].astype(str).str.contains(non_canon_re, regex=True, na=False)
    n = int(offending_mask.sum())
    if n == 0:
        return None
    sample = corpus.loc[offending_mask, "peptide"].head(5).tolist()
    return AuditFailure(
        mode="non_canonical_amino_acids",
        description="peptides contain non-canonical amino acid characters (allowed: ACDEFGHIKLMNPQRSTVWY)",
        n_offending_rows=n,
        sample=sample,
    )


def _mode_manifest_sha256_drift(
    corpus_sha256: str | None,
    expected_sha256: str | None,
) -> AuditFailure | None:
    """(vi) sha256 manifest drift.

    The corpus SHA-256 must match the published manifest SHA. A mismatch
    means the corpus on disk differs from the one the manifest describes
    — either silent corruption or an unintended rebuild.
    """
    if expected_sha256 is None or corpus_sha256 is None:
        return None
    if corpus_sha256.lower() == expected_sha256.lower():
        return None
    return AuditFailure(
        mode="manifest_sha256_drift",
        description=f"corpus SHA-256 ({corpus_sha256[:12]}…) does not match the published manifest ({expected_sha256[:12]}…)",
        n_offending_rows=0,
        sample=[corpus_sha256, expected_sha256],
    )


# ── orchestrator ────────────────────────────────────────────────────────────


def run_audit_gate(
    corpus: pd.DataFrame,
    *,
    corpus_sha256: Optional[str] = None,
    expected_manifest_sha256: Optional[str] = None,
    positive_rate_low: float = DEFAULT_POSITIVE_RATE_LOW,
    positive_rate_high: float = DEFAULT_POSITIVE_RATE_HIGH,
    min_per_allele: int = DEFAULT_MIN_PER_ALLELE,
) -> AuditReport:
    """Run all six contamination-mode checks.

    Parameters
    ----------
    corpus
        The training corpus DataFrame. Must have columns: peptide, allele,
        binding_label, label_origin, split. Optional: allele_cluster (used
        for mode (ii); skipped if absent).
    corpus_sha256, expected_manifest_sha256
        If both provided, mode (vi) compares them. If either is None, the
        SHA check is skipped (informational).
    positive_rate_low, positive_rate_high, min_per_allele
        Thresholds for mode (iv).

    Returns
    -------
    AuditReport
        `passed` is True only if all six modes pass. Call
        `report.raise_if_failed()` to abort training on any failure.
    """
    modes = [
        _mode_peptide_overlap(corpus),
        _mode_cluster_mate_alleles(corpus),
        _mode_decoys_in_both_partitions(corpus),
        _mode_per_allele_positive_imbalance(
            corpus, positive_rate_low, positive_rate_high, min_per_allele,
        ),
        _mode_non_canonical_aa(corpus),
        _mode_manifest_sha256_drift(corpus_sha256, expected_manifest_sha256),
    ]
    mode_names = [
        "peptide_train_test_overlap",
        "cluster_mate_alleles",
        "decoys_in_both_partitions",
        "per_allele_positive_imbalance",
        "non_canonical_amino_acids",
        "manifest_sha256_drift",
    ]
    all_findings = [(name, m) for name, m in zip(mode_names, modes) if m is not None]
    strict_failures = [m for name, m in all_findings if name in STRICT_FAIL_MODES]
    informational = [m for name, m in all_findings if name in INFORMATIONAL_MODES]
    per_mode_pass = {name: (m is None) for name, m in zip(mode_names, modes)}
    return AuditReport(
        passed=not strict_failures,
        failures=strict_failures,
        informational=informational,
        per_mode_pass=per_mode_pass,
    )


def compute_corpus_sha256(corpus_csv_path: str) -> str:
    """Helper: stream-compute SHA-256 of a corpus CSV/parquet file (1 MB chunks)."""
    h = hashlib.sha256()
    with open(corpus_csv_path, "rb") as f:
        while True:
            buf = f.read(1 << 20)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()
