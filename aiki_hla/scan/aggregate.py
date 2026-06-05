"""Stage 4 — scoring, aggregation, reporting.

Takes per-(peptide × allele) probabilities + the originating peptide
position list and computes:
  - per-residue hot-spot score (max over peptides covering each residue,
    weighted by allele population frequency)
  - contiguous hot-spot regions (≥5 consecutive residues above threshold)
  - top-K risky peptides ranked by aggregate-risk × allele coverage
  - per-allele summary (count of risky peptides per allele)
  - aggregate per-protein risk score (∈ [0, 1])

Population-frequency weighting in Phase 1 is uniform (each allele weight
1.0). Phase 2 introduces HLAfreq-driven weighting per region.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import pandas as pd

from aiki_hla.scan.alleles import AllelePanel
from aiki_hla.scan.extract import Peptide


# Default thresholds (configurable per scan). See plan §8.2.
#
# IMPORTANT — binder selection mode (validated on the influenza-M1 positive
# control, 2026-05-22): the deployed Class I model is overconfident
# (manuscript ECE 0.16), so an ABSOLUTE probability threshold flags an
# implausible fraction of proteome peptides (~93% of M1 9-mers at P>=0.5 for
# HLA-A*02:01). The model's RANK signal is excellent, though — the canonical
# GILGFVFTL epitope ranks 11/974 (99th pctile) for its restriction allele
# A*02:01 and 950/974 (2.6th pctile) for the negative-control B*07:02. We
# therefore default to PERCENTILE mode (top-N% per allele), matching the
# rank-percentile convention used by NetMHCpan/NetMHCIIpan. Absolute mode is
# retained for callers who have per-allele calibrated cutoffs.
DEFAULT_BINDER_MODE = "percentile"          # "percentile" | "absolute"
DEFAULT_BINDER_PERCENTILE = 2.0             # top 2% per allele = "binder" (NetMHCpan weak-binder rank)
DEFAULT_BINDER_THRESHOLD = 0.5              # only used in absolute mode
DEFAULT_HOTSPOT_THRESHOLD = 0.3             # per-residue cutoff, absolute mode
DEFAULT_HOTSPOT_PROMISCUITY = 0.50          # percentile mode: ≥50% of panel (weighted) binds → hot
DEFAULT_HOTSPOT_MIN_RUN = 5
DEFAULT_HOTSPOT_MERGE_GAP = 3


@dataclass
class Hotspot:
    """A contiguous run of residues above the hot-spot threshold."""
    start: int           # 1-indexed inclusive
    end: int             # 1-indexed inclusive
    length: int
    max_score: float
    mean_score: float
    contributing_alleles: tuple[str, ...]
    peptides: tuple[str, ...]


@dataclass
class RiskyPeptide:
    """A (peptide, allele) pair flagged as a binder above ``binder_threshold``."""
    peptide: str
    allele: str
    start: int            # 1-indexed position of the peptide's first residue in the source protein
    end: int
    length: int
    mhc_class: str
    binding_prob: float
    population_weighted_prob: float
    # Provenance flags (Phase 2 will populate: known_immunogenic_iedb, tolerogenic_likely).
    flags: tuple[str, ...] = field(default_factory=tuple)


@dataclass
class HotspotResult:
    """Aggregated scan output."""
    protein_length: int
    n_peptides_extracted: int
    n_scored_calls: int
    binder_mode: str                     # "percentile" | "absolute"
    binder_percentile: float             # top-N% per allele (percentile mode)
    binder_threshold: float              # binding_prob cutoff (absolute mode)
    hotspot_threshold: float
    per_residue_score: list[float]       # length == protein_length
    hotspots: list[Hotspot]
    risky_peptides: list[RiskyPeptide]
    per_allele_n_risky: dict[str, int]
    aggregate_risk: float
    allele_panel_name: str


def _detect_contiguous_runs(
    per_residue: list[float],
    threshold: float,
    min_run: int,
    merge_gap: int,
) -> list[tuple[int, int]]:
    """Return list of (start_1indexed, end_1indexed) contiguous runs above threshold.

    Adjacent runs separated by ``≤ merge_gap`` sub-threshold residues are merged.
    """
    L = len(per_residue)
    if L == 0:
        return []

    raw_runs: list[tuple[int, int]] = []
    i = 0
    while i < L:
        if per_residue[i] >= threshold:
            j = i
            while j < L and per_residue[j] >= threshold:
                j += 1
            if (j - i) >= min_run:
                raw_runs.append((i + 1, j))
            i = j
        else:
            i += 1

    if not raw_runs or merge_gap <= 0:
        return raw_runs

    merged: list[tuple[int, int]] = [raw_runs[0]]
    for s, e in raw_runs[1:]:
        ps, pe = merged[-1]
        if s - pe - 1 <= merge_gap:
            merged[-1] = (ps, e)
        else:
            merged.append((s, e))
    return merged


def aggregate_hotspots(
    *,
    protein_length: int,
    peptides: list[Peptide],
    scored_df: pd.DataFrame,
    panel: AllelePanel,
    binder_mode: str = DEFAULT_BINDER_MODE,
    binder_percentile: float = DEFAULT_BINDER_PERCENTILE,
    binder_threshold: float = DEFAULT_BINDER_THRESHOLD,
    hotspot_threshold: float | None = None,
    hotspot_min_run: int = DEFAULT_HOTSPOT_MIN_RUN,
    hotspot_merge_gap: int = DEFAULT_HOTSPOT_MERGE_GAP,
) -> HotspotResult:
    """Aggregate per-(peptide × allele) probabilities into a HotspotResult.

    Two binder-selection modes (see module header for why percentile is the
    default, validated on the influenza-M1 positive control):

      - ``"percentile"`` (default, calibration-robust): a (peptide, allele)
        is a binder iff its ``binding_prob`` is in the top ``binder_percentile``%
        of all peptides scored against that SAME allele (the NetMHCpan
        rank-percentile convention). The per-residue score is the max over
        covering pairs of ``weight × (1 - rank_fraction)``, so the heatmap is
        rank-based rather than raw-probability based.
      - ``"absolute"``: a binder iff ``binding_prob >= binder_threshold``;
        per-residue score is the max covering ``weight × binding_prob``.
        Use only with per-allele calibrated cutoffs.

    Args:
        protein_length: Length of the source protein (residue count).
        peptides: Stage-0 ``extract_peptides`` output (peptide → coords map).
        scored_df: Stage-3 output; columns ``peptide``, ``allele``, ``binding_prob``.
        panel: The AllelePanel (drives population weights).
        binder_mode: ``"percentile"`` or ``"absolute"``.
        binder_percentile: Top-N% per allele for percentile mode (default 2.0).
        binder_threshold: ``binding_prob`` cutoff for absolute mode.
        hotspot_threshold: Per-residue score cutoff for hot-spot detection.
            If None, defaults to ``1 - binder_percentile/100`` in percentile
            mode (residues covered by a top-percentile binder) or
            ``DEFAULT_HOTSPOT_THRESHOLD`` in absolute mode.
        hotspot_min_run: Minimum contiguous run length for a hot-spot region.
        hotspot_merge_gap: Adjacent hot-spots separated by ``≤ this``
            sub-threshold residues are merged.

    Returns:
        ``HotspotResult`` with per-residue scores, hot-spot regions, top
        risky peptides, per-allele counts, and aggregate-risk.
    """
    if binder_mode not in ("percentile", "absolute"):
        raise ValueError(f"binder_mode must be 'percentile' or 'absolute', got {binder_mode!r}")
    required = {"peptide", "allele", "binding_prob"}
    missing = required - set(scored_df.columns)
    if missing:
        raise ValueError(f"scored_df missing required columns: {missing}")

    if hotspot_threshold is None:
        # Percentile mode: per-residue score is the population-weighted fraction
        # of the panel presenting a binder (promiscuity). Default 0.50 means a
        # residue is "hot" if a MAJORITY of the panel (weighted) binds there.
        # This works across panel sizes: a 1-allele scan scores 0/1 per residue
        # (1.0 ≥ 0.50 → that allele's binding region is the hot-spot), while a
        # large panel surfaces only promiscuously-presented regions (the real
        # population-level immunogenicity risk; EpiVax ClustiMer intuition).
        hotspot_threshold = (
            DEFAULT_HOTSPOT_PROMISCUITY
            if binder_mode == "percentile"
            else DEFAULT_HOTSPOT_THRESHOLD
        )

    pep_positions: dict[str, list[tuple[int, int, int, str]]] = {}
    for p in peptides:
        pep_positions.setdefault(p.sequence, []).append(
            (p.start, p.end, p.length, p.mhc_class)
        )

    # Per-allele rank fraction (0 = strongest binder, ~1 = weakest) so that
    # binder selection and the per-residue signal are calibration-robust.
    scored_df = scored_df.copy()
    # rank_fraction = (rank-1)/(n-1); descending by prob → strongest = 0.0
    scored_df["_rank_frac"] = (
        scored_df.groupby("allele")["binding_prob"]
        .rank(ascending=False, method="first")
        .sub(1.0)
        .div(scored_df.groupby("allele")["binding_prob"].transform("count").sub(1.0).clip(lower=1.0))
    )

    per_residue = [0.0] * protein_length
    # Percentile mode: track the SET of binder alleles covering each residue so
    # the heatmap measures PROMISCUITY (how much of the HLA panel presents a
    # binder there — the EpiVax ClustiMer concept). A "max over alleles"
    # signal saturates with large panels because different alleles bind
    # different regions, so their union covers the whole protein.
    per_residue_alleles: list[set[str]] = [set() for _ in range(protein_length)]
    total_panel_weight = sum(panel.weight_for(a) for a in panel.alleles) or 1.0

    risky: list[RiskyPeptide] = []
    per_allele_n_risky: dict[str, int] = {a: 0 for a in panel.alleles}
    binder_pairs: set[tuple[str, str]] = set()

    for _, row in scored_df.iterrows():
        pep = row["peptide"]
        allele = row["allele"]
        prob = float(row["binding_prob"])
        rank_frac = float(row["_rank_frac"])
        rank_pct = rank_frac * 100.0
        weight = panel.weight_for(allele)

        if binder_mode == "percentile":
            is_binder = rank_pct <= binder_percentile
            residue_signal = 0.0  # promiscuity computed after the loop
        else:
            is_binder = prob >= binder_threshold
            residue_signal = weight * prob

        occurrences = pep_positions.get(pep, [])
        for start, end, length, mhc_class in occurrences:
            for r in range(start - 1, end):
                if r >= protein_length:
                    continue
                if binder_mode == "percentile":
                    if is_binder:
                        per_residue_alleles[r].add(allele)
                elif residue_signal > per_residue[r]:
                    per_residue[r] = residue_signal

            if is_binder:
                risky.append(
                    RiskyPeptide(
                        peptide=pep,
                        allele=allele,
                        start=start,
                        end=end,
                        length=length,
                        mhc_class=mhc_class,
                        binding_prob=prob,
                        population_weighted_prob=weight * prob,
                        flags=(f"rank_pctile={rank_pct:.2f}",),
                    )
                )
                per_allele_n_risky[allele] = per_allele_n_risky.get(allele, 0) + 1
                binder_pairs.add((pep, allele))

    if binder_mode == "percentile":
        # per_residue[r] = population-weighted fraction of the panel that
        # presents a top-percentile binder covering residue r. Range [0, 1].
        for r in range(protein_length):
            covered = sum(panel.weight_for(a) for a in per_residue_alleles[r])
            per_residue[r] = covered / total_panel_weight

    # Strongest binders first. binding_prob descending tracks per-allele rank
    # within an allele and is a consistent ordering across alleles for display.
    risky.sort(key=lambda r: r.binding_prob, reverse=True)

    # Hot-spot regions over the per-residue score.
    runs = _detect_contiguous_runs(
        per_residue,
        hotspot_threshold,
        hotspot_min_run,
        hotspot_merge_gap,
    )
    hotspots: list[Hotspot] = []
    for s, e in runs:
        region_scores = per_residue[s - 1:e]
        max_score = max(region_scores) if region_scores else 0.0
        mean_score = sum(region_scores) / len(region_scores) if region_scores else 0.0
        contributing_alleles, contributing_peptides = _peptides_alleles_in_region(
            peptides=peptides,
            region=(s, e),
            binder_pairs=binder_pairs,
        )
        hotspots.append(
            Hotspot(
                start=s,
                end=e,
                length=e - s + 1,
                max_score=max_score,
                mean_score=mean_score,
                contributing_alleles=contributing_alleles,
                peptides=contributing_peptides,
            )
        )

    aggregate_risk = sum(per_residue) / protein_length if protein_length else 0.0

    return HotspotResult(
        protein_length=protein_length,
        n_peptides_extracted=len(peptides),
        n_scored_calls=len(scored_df),
        binder_mode=binder_mode,
        binder_percentile=binder_percentile,
        binder_threshold=binder_threshold,
        hotspot_threshold=hotspot_threshold,
        per_residue_score=per_residue,
        hotspots=hotspots,
        risky_peptides=risky,
        per_allele_n_risky=per_allele_n_risky,
        aggregate_risk=aggregate_risk,
        allele_panel_name=panel.panel_name,
    )


def _peptides_alleles_in_region(
    *,
    peptides: list[Peptide],
    region: tuple[int, int],
    binder_pairs: set[tuple[str, str]],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Find the (alleles, peptides) whose binder pairs fall inside a region.

    A (peptide, allele) contributes iff the peptide overlaps the region AND
    the pair is in ``binder_pairs`` (the mode-agnostic binder set built by the
    caller — percentile-rank or absolute-threshold).
    """
    s, e = region
    peptides_in_region: set[str] = {
        p.sequence for p in peptides if p.start <= e and p.end >= s
    }
    alleles: set[str] = set()
    peps: set[str] = set()
    for pep, allele in binder_pairs:
        if pep in peptides_in_region:
            alleles.add(allele)
            peps.add(pep)
    return tuple(sorted(alleles)), tuple(sorted(peps))
