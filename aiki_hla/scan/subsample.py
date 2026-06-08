"""Uniform-stride sub-sampling helper for serving-environment pair-count caps.

When a scan request would produce more (peptide × allele) pairs than the
serving environment can complete under a latency budget, the scan computes
the smallest sliding-window stride that fits the pair count under the cap
and keeps every Nth peptide within each (class, length) group. The first and
last peptide of each group is always retained so the protein's termini are
represented.

This replaces the previous "chop the C-terminus" policy, which silently
left the back of the protein unscanned and hid any hot-spots there.
"""
from __future__ import annotations

import math
from collections import defaultdict

from aiki_hla.scan.extract import Peptide


def compute_stride(requested_pairs: int, max_pairs: int) -> int:
    """The smallest stride that brings the pair count under or equal to the cap."""
    if requested_pairs <= max_pairs:
        return 1
    return math.ceil(requested_pairs / max(max_pairs, 1))


def uniform_stride_subsample(
    peptides: list[Peptide],
    stride: int,
) -> list[Peptide]:
    """Keep every ``stride``-th peptide within each (class, length) group.

    The first and last peptide of each group are always retained so the
    protein's termini are represented (otherwise the rightmost peptide group
    might be silently dropped when its index is not a multiple of stride).

    Returns a new list; does not mutate the input.
    """
    if stride <= 1:
        return list(peptides)
    buckets: dict[tuple[str, int], list[Peptide]] = defaultdict(list)
    for p in peptides:
        buckets[(p.mhc_class, p.length)].append(p)
    kept: list[Peptide] = []
    for kind in sorted(buckets):
        grp = buckets[kind]
        idxs = list(range(0, len(grp), stride))
        if len(grp) > 1 and (len(grp) - 1) not in idxs:
            idxs.append(len(grp) - 1)
        for i in idxs:
            kept.append(grp[i])
    return kept


def fit_peptides_to_pair_budget(
    peptides: list[Peptide],
    n_alleles_per_class: dict[str, int],
    max_pairs: int,
) -> tuple[list[Peptide], int]:
    """Sub-sample peptides so the resulting pair count fits under ``max_pairs``.

    Args:
        peptides:               Stage-0 extracted peptide list.
        n_alleles_per_class:    {"I": n_ci, "II": n_cii} — used to estimate
                                pair count.
        max_pairs:              The serving-environment cap.

    Returns:
        (kept_peptides, stride). When stride > 1, ``kept_peptides`` is a
        uniformly sub-sampled copy of the input.
    """
    requested = sum(
        n_alleles_per_class.get(p.mhc_class, 0) for p in peptides
    )
    stride = compute_stride(requested, max_pairs)
    if stride == 1:
        return list(peptides), 1
    return uniform_stride_subsample(peptides, stride), stride
