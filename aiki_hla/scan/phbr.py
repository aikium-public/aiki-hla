"""Stage 7 — Patient HLA Binding Rank (PHBR).

Marty *et al.*, *Cell* 171, 1272 (2017). For each candidate peptide and the
patient's specific HLA-I (up to 6) or HLA-II (up to 12, including
DPA1-DPB1 / DQA1-DQB1 heterodimer combinations) alleles, take the **best**
percentile rank across the patient's alleles. Top neoantigens are those
with PHBR < 2%.

Field convention:

  PHBR-I  = min over patient's HLA-A/B/C alleles of (per-allele rank
            percentile of the peptide's binding score in the panel's
            distribution).
  PHBR-II = same but over patient's HLA-DR/DP/DQ alleles.

A peptide with PHBR < 2% is presented by at least one of the patient's
HLAs (the conventional weak-binder threshold). PHBR < 0.5% = strong
binder presented by at least one patient HLA.
"""
from __future__ import annotations

import pandas as pd


def compute_phbr(
    scored_df: pd.DataFrame,
    patient_alleles: list[str],
    *,
    score_col: str = "binding_prob",
) -> pd.DataFrame:
    """Compute PHBR per peptide across the patient's HLA set.

    Args:
        scored_df:        DataFrame with columns ``peptide``, ``allele``,
                          ``score_col`` (default ``binding_prob``).
        patient_alleles:  List of HLA allele names matching scored_df's
                          allele column. PHBR is computed across these only.
        score_col:        Column holding the binding probability (or
                          composite score) per (peptide, allele).

    Returns:
        DataFrame with columns:
          peptide:         the peptide string
          phbr_score:      best (highest) score across patient's alleles
          phbr_pctile:     rank percentile of phbr_score within the FULL
                           scored_df's per-allele distributions (i.e., 100 ×
                           rank / n_peptides for each allele, then min).
          phbr_allele:     which patient HLA produced the best score
          n_patient_alleles_with_strong_binder:  count of patient HLAs
                           with this peptide at < 0.5% rank (strong binder)
          n_patient_alleles_with_weak_binder:    count at < 2% rank (weak)

    A peptide is a candidate neoantigen for this patient iff its
    ``phbr_pctile < 2.0`` (weak) or ``< 0.5`` (strong) — the conventional
    NetMHCpan binder thresholds.
    """
    if not patient_alleles:
        raise ValueError("patient_alleles must be a non-empty list")
    missing_cols = {"peptide", "allele", score_col} - set(scored_df.columns)
    if missing_cols:
        raise ValueError(f"scored_df missing required columns: {missing_cols}")

    # Rank percentile per allele within the FULL scored_df (so the patient's
    # rank is comparable to NetMHCpan EL %rank computed against the panel's
    # distribution). HIGHER score → LOWER rank (stronger binder).
    df = scored_df[["peptide", "allele", score_col]].copy()
    n_per_allele = df.groupby("allele")["peptide"].transform("count")
    rank_desc = df.groupby("allele")[score_col].rank(method="min", ascending=False)
    df["_rank_pctile"] = (rank_desc / n_per_allele) * 100.0

    rel = df[df["allele"].isin(patient_alleles)].copy()
    if rel.empty:
        raise ValueError(
            f"none of the patient_alleles {patient_alleles} appear in scored_df"
        )

    # Take per peptide the best (smallest) rank percentile across patient
    # alleles. The score at that rank is the PHBR score; the allele is the
    # PHBR allele.
    rel = rel.sort_values("_rank_pctile").reset_index(drop=True)
    best_per_peptide = rel.drop_duplicates(subset=["peptide"], keep="first")
    out = best_per_peptide[["peptide", "allele", score_col, "_rank_pctile"]].rename(
        columns={
            "allele":      "phbr_allele",
            score_col:     "phbr_score",
            "_rank_pctile": "phbr_pctile",
        }
    )

    # Per-peptide counts at the two conventional binder thresholds.
    strong = rel[rel["_rank_pctile"] < 0.5].groupby("peptide").size().rename("n_strong")
    weak   = rel[rel["_rank_pctile"] < 2.0].groupby("peptide").size().rename("n_weak")
    out = out.merge(strong, on="peptide", how="left")
    out = out.merge(weak,   on="peptide", how="left")
    out["n_patient_alleles_with_strong_binder"] = out["n_strong"].fillna(0).astype(int)
    out["n_patient_alleles_with_weak_binder"]   = out["n_weak"].fillna(0).astype(int)
    out = out.drop(columns=["n_strong", "n_weak"])

    return out.reset_index(drop=True)


def is_neoantigen_candidate(
    phbr_pctile: float,
    *,
    threshold: float = 2.0,
) -> bool:
    """A peptide is a candidate neoantigen iff PHBR < threshold.

    Default ``threshold = 2.0`` matches the NetMHCpan weak-binder cutoff.
    Use ``threshold = 0.5`` for strong-binder-only ranking.
    """
    return phbr_pctile < threshold
