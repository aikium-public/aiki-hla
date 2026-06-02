"""Evaluation utilities for AIKI-HLA.

Public surface:
    compute_tiered_metrics   bootstrap-CI'd per-allele AUROC across the four
                             evaluation strata (broad, novel-peptide,
                             novel-allele, both-novel) plus the strict
                             novel-allele 9-allele headline.

The 9-allele strict-stratum allele set is bundled at
`aiki_hla/data/strict_novel_allele_stratum.json`.
"""
from aiki_hla.evaluation.tiered_metrics import compute_tiered_metrics  # noqa: F401

__all__ = ["compute_tiered_metrics"]
