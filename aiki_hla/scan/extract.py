"""Stage 0 — peptide extraction.

Sliding-window peptide extraction from a protein sequence. Each peptide is
emitted with its 1-indexed start position, end position, length, and
intended MHC class. Default lengths follow the AIKI-HLA training corpus
peptide-length distribution: Class I 8-11 mers, Class II 13-17 mers
(see docs/immunogenicity_screening_tool_plan.md §4).
"""
from __future__ import annotations

from dataclasses import dataclass


CLASS_I_LENGTHS: tuple[int, ...] = (8, 9, 10, 11)
CLASS_II_LENGTHS: tuple[int, ...] = (13, 14, 15, 16, 17)

CANONICAL_AAS = set("ACDEFGHIKLMNPQRSTVWY")


@dataclass(frozen=True)
class Peptide:
    """A single extracted peptide with its source-protein coordinates."""
    sequence: str
    start: int       # 1-indexed inclusive
    end: int         # 1-indexed inclusive
    length: int
    mhc_class: str   # "I" or "II"

    def __post_init__(self) -> None:
        if self.length != len(self.sequence):
            raise ValueError(
                f"Peptide length {self.length} disagrees with sequence "
                f"length {len(self.sequence)} for {self.sequence!r}"
            )
        if self.end - self.start + 1 != self.length:
            raise ValueError(
                f"Peptide span {self.start}..{self.end} disagrees with "
                f"length {self.length}"
            )
        if self.mhc_class not in ("I", "II"):
            raise ValueError(f"mhc_class must be 'I' or 'II', got {self.mhc_class!r}")


def _clean_sequence(seq: str) -> str:
    """Strip whitespace and uppercase a protein sequence; do not remove unknown residues."""
    return "".join(seq.split()).upper()


def _validate_canonical(seq: str) -> tuple[str, list[int]]:
    """Return (cleaned, indices_of_non_canonical_residues_1_indexed)."""
    cleaned = _clean_sequence(seq)
    bad = [i + 1 for i, aa in enumerate(cleaned) if aa not in CANONICAL_AAS]
    return cleaned, bad


def extract_peptides(
    sequence: str,
    mhc_class: str = "both",
    lengths_class_i: tuple[int, ...] = CLASS_I_LENGTHS,
    lengths_class_ii: tuple[int, ...] = CLASS_II_LENGTHS,
    skip_non_canonical: bool = True,
    region: tuple[int, int] | None = None,
) -> list[Peptide]:
    """Extract overlapping k-mer peptides from a protein sequence.

    Args:
        sequence: The protein amino-acid sequence (any case, whitespace OK).
        mhc_class: "I", "II", or "both". Determines which lengths to use.
        lengths_class_i: Tuple of peptide lengths to extract for Class I.
        lengths_class_ii: Tuple of peptide lengths to extract for Class II.
        skip_non_canonical: If True, drop peptides containing any character
            outside the 20 canonical amino acids (e.g. selenocysteine 'U',
            stop codons '*', ambiguous 'X', 'B', 'Z'). If False, emit them
            anyway and rely on downstream scoring to handle them.
        region: Optional (start, end) 1-indexed inclusive range to restrict
            extraction. Useful for scanning CDRs or surface-exposed regions
            only. If None, the full sequence is scanned.

    Returns:
        List of Peptide records, sorted by (start, length, mhc_class).
        Peptides are NOT de-duplicated; an identical substring at two
        different positions yields two records (for per-residue
        position-aware aggregation downstream). For de-duplicated string
        extraction call ``unique_peptide_strings(peptides)``.

    Notes:
        Class I and Class II have different binding-pocket biology and
        different conventional peptide-length distributions. The AIKI-HLA
        model accepts peptide lengths 8-25 across both classes; the
        defaults here are the field-conventional ranges (Class I 8-11,
        Class II 13-17). Pass explicit ``lengths_*`` tuples to override.
    """
    if mhc_class not in ("I", "II", "both"):
        raise ValueError(f"mhc_class must be 'I', 'II', or 'both', got {mhc_class!r}")

    cleaned = _clean_sequence(sequence)
    if not cleaned:
        return []

    if region is not None:
        start, end = region
        if start < 1 or end > len(cleaned) or start > end:
            raise ValueError(
                f"region {region} out of bounds for sequence of length {len(cleaned)}"
            )
        cleaned = cleaned[start - 1:end]
        offset = start - 1
    else:
        offset = 0

    classes_to_run: list[tuple[str, tuple[int, ...]]] = []
    if mhc_class in ("I", "both"):
        classes_to_run.append(("I", lengths_class_i))
    if mhc_class in ("II", "both"):
        classes_to_run.append(("II", lengths_class_ii))

    out: list[Peptide] = []
    for cls, lengths in classes_to_run:
        for k in lengths:
            if k <= 0 or k > len(cleaned):
                continue
            for i in range(len(cleaned) - k + 1):
                window = cleaned[i:i + k]
                if skip_non_canonical and not all(a in CANONICAL_AAS for a in window):
                    continue
                out.append(
                    Peptide(
                        sequence=window,
                        start=offset + i + 1,
                        end=offset + i + k,
                        length=k,
                        mhc_class=cls,
                    )
                )

    out.sort(key=lambda p: (p.start, p.length, p.mhc_class))
    return out


def unique_peptide_strings(peptides: list[Peptide]) -> list[str]:
    """Return the unique peptide strings (preserving first-seen order)."""
    seen: set[str] = set()
    out: list[str] = []
    for p in peptides:
        if p.sequence not in seen:
            seen.add(p.sequence)
            out.append(p.sequence)
    return out
