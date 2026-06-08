"""Stage 8 — Differential Agretopicity Index (DAI).

Duan *et al.*, *Nat Genet* 46, 1267 (2014). For a somatic missense mutation,
the DAI compares the MHC presentation of the mutant peptide to the
wild-type peptide carrying the same anchor positions. A LARGER DAI means
the mutant is more visible to the immune system than the parental sequence
— the field-standard signal for an immunogenic neoantigen.

Two conventions in the literature:

  log2 form (preferred for unbounded ratios):
      DAI = log2( p_MT / p_WT )

  difference form (simpler, more common in legacy tables):
      DAI = score_MT − score_WT

This module exposes both. Default = ``log2`` form because (a) the field has
been moving toward it since Łuksza 2017, and (b) it is symmetric around 0
which makes the "mutant-vs-WT advantage" comparable across alleles.

We also expose ``mutation_spec_to_peptides`` which parses the canonical
``G12D`` and ``12:G>D`` mutation-spec notations and emits the (WT, MT)
peptide pairs covering the mutation site, for both Class I and Class II
core lengths.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

import pandas as pd

from aiki_hla.scan.mutations import (
    AAS,
    CANON,
    CI_LEN_DEFAULT,
    CII_LEN_DEFAULT,
    _covering_window_starts,
)

_CANON_RE  = re.compile(r"^([ACDEFGHIKLMNPQRSTVWY])(\d+)([ACDEFGHIKLMNPQRSTVWY])$")
_COLON_RE  = re.compile(r"^(\d+):([ACDEFGHIKLMNPQRSTVWY])>([ACDEFGHIKLMNPQRSTVWY])$")


@dataclass
class MutationSpec:
    """A parsed somatic missense mutation against a 1-based sequence index."""
    position: int                 # 1-based
    wt_aa: str
    mt_aa: str

    def __post_init__(self) -> None:
        if self.wt_aa not in CANON or self.mt_aa not in CANON:
            raise ValueError(f"non-canonical amino acid in spec: {self}")
        if self.wt_aa == self.mt_aa:
            raise ValueError(f"WT and MT amino acids identical at position {self.position}")


def parse_mutation_spec(spec: str) -> MutationSpec:
    """Parse ``G12D`` (canonical AaposBbb) or ``12:G>D`` notation.

    Raises ValueError on any other format.
    """
    s = spec.strip().upper()
    m = _CANON_RE.match(s)
    if m:
        return MutationSpec(position=int(m.group(2)), wt_aa=m.group(1), mt_aa=m.group(3))
    m = _COLON_RE.match(s)
    if m:
        return MutationSpec(position=int(m.group(1)), wt_aa=m.group(2), mt_aa=m.group(3))
    raise ValueError(
        f"could not parse mutation spec {spec!r}; expected forms 'G12D' or '12:G>D'"
    )


def apply_mutation(sequence: str, mut: MutationSpec) -> str:
    """Apply ``mut`` to ``sequence``, asserting that the WT residue matches.

    Raises ValueError if the WT amino acid at the spec's position does NOT
    match the sequence — this catches the common error of pasting the wrong
    reference protein and reporting a misleading DAI.
    """
    if not (1 <= mut.position <= len(sequence)):
        raise ValueError(
            f"mutation position {mut.position} out of bounds for length-{len(sequence)} sequence"
        )
    actual = sequence[mut.position - 1]
    if actual != mut.wt_aa:
        raise ValueError(
            f"WT mismatch: spec says {mut.wt_aa!r} at position {mut.position}, "
            f"sequence has {actual!r}"
        )
    return sequence[: mut.position - 1] + mut.mt_aa + sequence[mut.position:]


def mutation_spec_to_peptides(
    sequence: str,
    mut: MutationSpec,
    mhc_class: str = "I",
    *,
    ci_len: int = CI_LEN_DEFAULT,
    cii_len: int = CII_LEN_DEFAULT,
) -> list[dict]:
    """Enumerate (WT, MT) peptide pairs covering the mutation site.

    Returns a list of dicts:
        {"wt_peptide": str, "mt_peptide": str, "start_1based": int,
         "length": int, "mhc_class": "I" | "II", "mutation_offset_1based": int}

    The ``mutation_offset_1based`` is the 1-based index of the mutated
    residue WITHIN the peptide — useful for downstream anchor-residue
    analysis (the field's notation is "P3" = third residue of a 9mer).
    """
    seq_len = len(sequence)
    mt_seq  = apply_mutation(sequence, mut)
    out: list[dict] = []

    def _emit(core_len: int, mhc_cls: str) -> None:
        for s in _covering_window_starts(seq_len, mut.position, core_len):
            wt_pep = sequence[s : s + core_len]
            mt_pep = mt_seq  [s : s + core_len]
            if set(wt_pep) <= CANON and set(mt_pep) <= CANON:
                out.append({
                    "wt_peptide":              wt_pep,
                    "mt_peptide":              mt_pep,
                    "start_1based":            s + 1,
                    "length":                  core_len,
                    "mhc_class":               mhc_cls,
                    "mutation_offset_1based":  mut.position - s,
                })

    if mhc_class in ("I", "both"):
        _emit(ci_len, "I")
    if mhc_class in ("II", "both"):
        _emit(cii_len, "II")
    if not out:
        raise ValueError(
            f"no valid covering peptides for {mut} in a length-{seq_len} sequence"
        )
    return out


def differential_agretopicity(
    p_mt: float,
    p_wt: float,
    *,
    mode: str = "log2",
    epsilon: float = 1e-6,
) -> float:
    """Compute DAI for one (WT, MT) score pair.

    Args:
        p_mt:    Mutant peptide presentation probability.
        p_wt:    Wild-type peptide presentation probability.
        mode:    "log2" (default — Łuksza-style ratio) or "diff" (legacy).
        epsilon: Floor on probabilities for log2 numerical stability.

    Returns:
        float DAI. Positive = mutant more visible than WT (immunogenic
        direction); negative = mutant less visible.
    """
    if mode == "log2":
        return math.log2(max(p_mt, epsilon) / max(p_wt, epsilon))
    if mode == "diff":
        return p_mt - p_wt
    raise ValueError(f"mode must be 'log2' or 'diff'; got {mode!r}")


def score_neoantigens(
    sequence: str,
    mutations: list[str | MutationSpec],
    scored_df: pd.DataFrame,
    *,
    patient_alleles: list[str] | None = None,
    mhc_class: str = "I",
    dai_mode: str = "log2",
    ci_len: int = CI_LEN_DEFAULT,
    cii_len: int = CII_LEN_DEFAULT,
    score_col: str = "p_composite",
) -> pd.DataFrame:
    """Rank candidate neoantigens for one patient against a list of mutations.

    For each mutation × allele × covering peptide:
      1. Score WT and MT peptides via the pre-computed ``scored_df`` lookup.
      2. Compute DAI = differential_agretopicity(p_MT, p_WT, mode=dai_mode).
      3. If ``patient_alleles`` is supplied, also report per-mutation PHBR
         of the MT peptide across the patient's alleles.

    Args:
        sequence:         Wild-type protein sequence.
        mutations:        List of mutation specs (strings or MutationSpec).
        scored_df:        Pre-scored (peptide, allele, p_composite) table
                          covering ALL the (WT, MT) peptides for these
                          mutations.
        patient_alleles:  Optional patient HLA set for PHBR.
        mhc_class:        "I", "II", or "both".
        dai_mode:         "log2" (default) or "diff".
        ci_len:           Class I core length (default 9).
        cii_len:          Class II core length (default 15).
        score_col:        Column in scored_df holding p_composite-like score.

    Returns:
        DataFrame sorted by DAI descending (most immunogenic first), with
        columns:
          mutation, allele, wt_peptide, mt_peptide, start_1based,
          mutation_offset_1based, length, mhc_class, p_wt, p_mt, dai,
          dai_mode, [phbr_pctile if patient_alleles given]
    """
    parsed: list[MutationSpec] = [
        m if isinstance(m, MutationSpec) else parse_mutation_spec(m)
        for m in mutations
    ]
    lookup = {
        (row["peptide"], row["allele"]): row[score_col]
        for _, row in scored_df.iterrows()
    }

    rows: list[dict] = []
    for mut in parsed:
        pep_pairs = mutation_spec_to_peptides(
            sequence, mut, mhc_class=mhc_class, ci_len=ci_len, cii_len=cii_len,
        )
        alleles_in_table = scored_df["allele"].unique()
        for pp in pep_pairs:
            for allele in alleles_in_table:
                p_wt = lookup.get((pp["wt_peptide"], allele))
                p_mt = lookup.get((pp["mt_peptide"], allele))
                if p_wt is None or p_mt is None:
                    continue
                dai = differential_agretopicity(p_mt, p_wt, mode=dai_mode)
                rows.append({
                    "mutation":              f"{mut.wt_aa}{mut.position}{mut.mt_aa}",
                    "allele":                allele,
                    "wt_peptide":            pp["wt_peptide"],
                    "mt_peptide":            pp["mt_peptide"],
                    "start_1based":          pp["start_1based"],
                    "mutation_offset_1based": pp["mutation_offset_1based"],
                    "length":                pp["length"],
                    "mhc_class":             pp["mhc_class"],
                    "p_wt":                  round(p_wt, 4),
                    "p_mt":                  round(p_mt, 4),
                    "dai":                   round(dai, 4),
                    "dai_mode":              dai_mode,
                })

    if not rows:
        return pd.DataFrame(columns=[
            "mutation", "allele", "wt_peptide", "mt_peptide", "start_1based",
            "mutation_offset_1based", "length", "mhc_class",
            "p_wt", "p_mt", "dai", "dai_mode",
        ])

    out = pd.DataFrame(rows).sort_values("dai", ascending=False).reset_index(drop=True)

    if patient_alleles:
        # PHBR computed on the MT peptides across patient's alleles, using the
        # SAME scored_df as the rank distribution.
        from aiki_hla.scan.phbr import compute_phbr
        mt_scored = scored_df[scored_df["peptide"].isin(out["mt_peptide"].unique())].copy()
        mt_scored = mt_scored.rename(columns={score_col: "binding_prob"})
        phbr = compute_phbr(mt_scored, patient_alleles)
        out = out.merge(
            phbr[["peptide", "phbr_pctile", "phbr_allele"]].rename(
                columns={"peptide": "mt_peptide"}
            ),
            on="mt_peptide",
            how="left",
        )

    return out
