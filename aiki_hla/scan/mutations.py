"""Stage 6 — Per-position deimmunization / mutation sweep.

Given a position in a protein, score all 20 amino acid substitutions at that
position against a chosen allele panel and report which substitutions LOWER
the local hot-spot (deimmunization view) or RAISE binding vs wild-type
(neoantigen-visibility view; this is the field-standard differential
agretopicity intuition applied to a single-AA scan).

Per-allele aggregation: MAX over covering peptides (the NetMHCpan / EpiVax /
IEDB convention — a single covering binder is enough to immunize, so
averaging would dilute the signal). Cross-allele aggregation: MAX (worst
case) plus a promiscuity count (fraction of panel alleles with a
strong binder).

The module is callable-agnostic: the caller supplies a `score_fn` that takes
a list of (peptide, allele) dicts and returns scored rows. This way the
same logic powers:

  - Modal: `score_fn = AikiMhcModel().score_pairs.remote`
  - pip CLI: `score_fn = lambda pairs: inference.score_dataframe(pairs, ...)`
  - tests:  `score_fn = mock returning deterministic values`

The function does not know about ESM-2, torch, or Modal; it is pure Python.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Sequence

from aiki_hla.scan.alleles import AllelePanel

AAS = "ACDEFGHIKLMNPQRSTVWY"            # 20 canonical amino acids
CANON = set(AAS)
CI_LEN_DEFAULT = 9                       # canonical NetMHCpan Class I core
CII_LEN_DEFAULT = 15                     # canonical NetMHCIIpan Class II core


@dataclass
class SubstitutionResult:
    """One substituted amino-acid's worth of evaluation at the target position."""
    aa: str
    is_wild_type: bool
    max_p_composite: float               # worst-case across (covering pep × allele)
    worst_allele: str | None
    worst_peptide: str | None
    promiscuity: float                   # fraction of panel with a strong (≥0.5) binder
    n_alleles_strong: int
    n_alleles_total: int
    delta_vs_wt: float = 0.0             # max_p_composite − wt_max_p_composite


@dataclass
class MutationSweepResult:
    """A complete 20-AA sweep at a single position."""
    sequence: str                        # the (unsubstituted) input sequence
    position: int                        # 1-based residue index
    wild_type_aa: str
    wild_type_max_p_composite: float
    mhc_class: str                       # "I" | "II" | "both"
    panel_alleles: tuple[str, ...]
    core_lengths: dict[str, int | None]
    n_pairs_scored: int
    results: list[SubstitutionResult]    # 20 entries, indexed by AAS order


def _covering_window_starts(seq_len: int, pos1: int, core_len: int) -> list[int]:
    """Yield 0-based starts whose [start, start+core_len) window includes pos1 (1-based)."""
    lo = max(0, pos1 - core_len)
    hi = min(seq_len - core_len + 1, pos1)
    return list(range(lo, hi))


def _split_panel_by_class(panel: AllelePanel) -> tuple[list[str], list[str]]:
    """Split the resolved panel into Class I + Class II allele lists."""
    cii_tokens = ("DR", "DP", "DQ")
    class_ii = [a for a in panel.alleles if any(t in a for t in cii_tokens)]
    class_i  = [a for a in panel.alleles if a not in class_ii]
    return class_i, class_ii


def score_mutations(
    *,
    sequence: str,
    position: int,                       # 1-based
    mhc_class: str,                      # "I" | "II" | "both"
    panel: AllelePanel,
    score_fn: Callable[[list[dict]], list[dict]],
    ci_len: int = CI_LEN_DEFAULT,
    cii_len: int = CII_LEN_DEFAULT,
    max_pairs: int | None = 4000,
) -> MutationSweepResult:
    """Sweep all 20 amino acid substitutions at ``position`` and aggregate.

    Args:
        sequence:    The unsubstituted (wild-type or background) sequence.
        position:    1-based residue index to mutate.
        mhc_class:   "I", "II", or "both" — selects which class's core length
                     is scored.
        panel:       Resolved AllelePanel (from `resolve_allele_panel`).
        score_fn:    Callable taking a list of {peptide, allele} dicts and
                     returning scored rows with `p_aiki`, `p_gate`,
                     `p_composite` (i.e. AIKI mode="both" output shape).
        ci_len:      Class I core length (default 9).
        cii_len:     Class II core length (default 15).
        max_pairs:   Optional cap on the total unique (peptide, allele) pair
                     count for the sweep. Raises ``ValueError`` if exceeded;
                     None disables the check.

    Returns:
        A :class:`MutationSweepResult` with 20 entries sorted in the
        canonical AAS order (alphabetical by amino-acid code).

    Raises:
        ValueError: on invalid position, mhc_class, or pair-count over cap.
    """
    seq_len = len(sequence)
    if not (1 <= position <= seq_len):
        raise ValueError(f"position {position} out of bounds for length-{seq_len} sequence")
    if mhc_class not in ("I", "II", "both"):
        raise ValueError(f"mhc_class must be 'I', 'II', or 'both'; got {mhc_class!r}")

    class_i_alleles, class_ii_alleles = _split_panel_by_class(panel)
    wt_aa = sequence[position - 1]

    # Build all unique (peptide, allele) pairs across the 20 substitutions.
    all_pairs_set: set[tuple[str, str]] = set()
    per_aa_pairs: dict[str, list[tuple[str, str]]] = {}

    for aa in AAS:
        mut_seq = sequence[: position - 1] + aa + sequence[position:]
        pairs: list[tuple[str, str]] = []
        if mhc_class in ("I", "both") and class_i_alleles:
            for s in _covering_window_starts(seq_len, position, ci_len):
                pep = mut_seq[s : s + ci_len]
                if set(pep) <= CANON:
                    for allele in class_i_alleles:
                        pairs.append((pep, allele))
                        all_pairs_set.add((pep, allele))
        if mhc_class in ("II", "both") and class_ii_alleles:
            for s in _covering_window_starts(seq_len, position, cii_len):
                pep = mut_seq[s : s + cii_len]
                if set(pep) <= CANON:
                    for allele in class_ii_alleles:
                        pairs.append((pep, allele))
                        all_pairs_set.add((pep, allele))
        per_aa_pairs[aa] = pairs

    if not all_pairs_set:
        raise ValueError(f"no valid covering peptides at position {position}")
    if max_pairs is not None and len(all_pairs_set) > max_pairs:
        raise ValueError(
            f"position {position} would need {len(all_pairs_set):,} pair scores; "
            f"cap is {max_pairs:,}"
        )

    pair_list = [{"peptide": p, "allele": a} for (p, a) in all_pairs_set]
    scored = score_fn(pair_list)
    lookup: dict[tuple[str, str], dict[str, Any]] = {
        (r["peptide"], r["allele"]): r for r in scored if "error" not in r
    }

    # Per-AA aggregation: per-allele MAX p_composite over covering peptides.
    wt_max: float | None = None
    results: list[SubstitutionResult] = []
    for aa in AAS:
        per_allele_best: dict[str, float] = {}
        best_pep_for_allele: dict[str, str] = {}
        for pep, allele in per_aa_pairs[aa]:
            r = lookup.get((pep, allele))
            if r is None:
                continue
            comp = float(r.get("p_composite", 0.0))
            if comp > per_allele_best.get(allele, -1.0):
                per_allele_best[allele] = comp
                best_pep_for_allele[allele] = pep
        if per_allele_best:
            worst_allele = max(per_allele_best, key=per_allele_best.get)
            max_comp = per_allele_best[worst_allele]
            worst_pep = best_pep_for_allele[worst_allele]
            n_strong = sum(1 for v in per_allele_best.values() if v >= 0.5)
            promiscuity = n_strong / len(per_allele_best)
        else:
            worst_allele = None
            max_comp = 0.0
            worst_pep = None
            n_strong = 0
            promiscuity = 0.0
        results.append(SubstitutionResult(
            aa=aa,
            is_wild_type=(aa == wt_aa),
            max_p_composite=round(max_comp, 4),
            worst_allele=worst_allele,
            worst_peptide=worst_pep,
            promiscuity=round(promiscuity, 4),
            n_alleles_strong=n_strong,
            n_alleles_total=len(per_allele_best),
        ))
        if aa == wt_aa:
            wt_max = max_comp

    # delta_vs_wt: NEGATIVE = improvement (substitution kills the local hot-spot);
    # POSITIVE = regression (substitution makes the local binder stronger).
    for r in results:
        r.delta_vs_wt = round(r.max_p_composite - (wt_max or 0.0), 4)

    return MutationSweepResult(
        sequence=sequence,
        position=position,
        wild_type_aa=wt_aa,
        wild_type_max_p_composite=round(wt_max or 0.0, 4),
        mhc_class=mhc_class,
        panel_alleles=tuple(panel.alleles),
        core_lengths={
            "class_I":  ci_len  if mhc_class in ("I", "both")  else None,
            "class_II": cii_len if mhc_class in ("II", "both") else None,
        },
        n_pairs_scored=len(pair_list),
        results=results,
    )


def best_deimmunization(result: MutationSweepResult) -> SubstitutionResult | None:
    """Return the single substitution that most lowers the local hot-spot.

    Returns None if no non-wild-type substitution improves on the wild-type
    score (every substitution is equal or worse, in which case the local
    epitope is not amenable to point-mutation deimmunization at this position
    alone).
    """
    improvers = [r for r in result.results if not r.is_wild_type and r.delta_vs_wt < 0]
    if not improvers:
        return None
    return min(improvers, key=lambda r: r.delta_vs_wt)


def best_neoantigen_visibility(result: MutationSweepResult) -> SubstitutionResult | None:
    """Return the single substitution that most RAISES binding vs wild-type.

    The mirror image of :func:`best_deimmunization`. In a cancer-vaccine
    framing this is the substitution that would produce the strongest
    putative neoantigen candidate (largest positive delta_vs_wt). Use with
    care: this is single-AA scan output, not a full DAI computation against
    a real somatic mutation. For the real DAI workflow see :mod:`dai`.

    Returns None if no substitution improves on the wild-type score.
    """
    visibles = [r for r in result.results if not r.is_wild_type and r.delta_vs_wt > 0]
    if not visibles:
        return None
    return max(visibles, key=lambda r: r.delta_vs_wt)
