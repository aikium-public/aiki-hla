"""Stage 9 — Long Synthetic Peptide (LSP) construction.

The BioNTech/Moderna mRNA-vaccine convention is to encode each neoantigen
candidate as a 25–30mer "long synthetic peptide" centred on the predicted
strong-binder core epitope, with native flanks on both sides. The LSP gives
the antigen-presenting cell flexibility to process the peptide into both
Class I and Class II epitopes via proteasomal and endolysosomal pathways
respectively (Sahin *et al.*, *Nature* 547, 222 (2017); Ott *et al.*,
*Nature* 547, 217 (2017); Hu *et al.*, *Nature* 565, 234 (2019); Rojas
*et al.*, *Nature* 618, 144 (2023)).

This module computes the LSP coordinates given:

  - the wild-type / source protein sequence,
  - the core epitope's start and length (or the core peptide itself),
  - the target LSP length (default 27 — the BioNTech BNT122 / autogene
    cevumeran convention; Moderna mRNA-4157 uses ~25-mers).

Truncation rules at protein termini are explicit and recorded in the
output for downstream review.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LSPDesign:
    """Coordinates of one long synthetic peptide design."""
    core_epitope:            str
    core_start_1based:       int      # start of the core in the protein
    core_end_1based:         int      # end of the core in the protein (inclusive)
    lsp_sequence:            str
    lsp_start_1based:        int
    lsp_end_1based:          int
    target_length:           int
    actual_length:           int
    n_flank_truncated:       int      # how many residues couldn't be added at N-terminus
    c_flank_truncated:       int      # how many residues couldn't be added at C-terminus


def build_lsp(
    protein_sequence: str,
    core_start_1based: int,
    core_length: int,
    *,
    lsp_length: int = 27,
) -> LSPDesign:
    """Build a long synthetic peptide centred on the core epitope.

    Args:
        protein_sequence:    Wild-type or background protein sequence.
        core_start_1based:   1-based start of the core epitope in the protein.
        core_length:         Length of the core epitope (typically 9 for
                             Class I, 15 for Class II).
        lsp_length:          Target LSP length (default 27; BioNTech BNT122
                             autogene cevumeran convention).

    Returns:
        :class:`LSPDesign` with full coordinates and any truncation counts.

    Raises:
        ValueError: if ``core_start_1based`` or ``core_length`` would extend
                    past the protein, or if ``lsp_length < core_length``.
    """
    seq_len = len(protein_sequence)
    core_end_1based = core_start_1based + core_length - 1
    if core_start_1based < 1 or core_end_1based > seq_len:
        raise ValueError(
            f"core [{core_start_1based}, {core_end_1based}] out of bounds "
            f"for length-{seq_len} sequence"
        )
    if lsp_length < core_length:
        raise ValueError(
            f"lsp_length ({lsp_length}) must be ≥ core_length ({core_length})"
        )

    flank_total = lsp_length - core_length
    # Symmetric flanks; if odd, give the extra residue to the C-terminus
    # (more space for proteasome cleavage past the C-anchor).
    n_flank_target = flank_total // 2
    c_flank_target = flank_total - n_flank_target

    lsp_start_1based = core_start_1based - n_flank_target
    lsp_end_1based   = core_end_1based   + c_flank_target

    n_truncated = 0
    c_truncated = 0
    if lsp_start_1based < 1:
        n_truncated = 1 - lsp_start_1based
        lsp_start_1based = 1
        # Try to recover length on the C-terminal side.
        extra_c = min(n_truncated, seq_len - lsp_end_1based)
        lsp_end_1based += extra_c
        n_truncated -= extra_c
    if lsp_end_1based > seq_len:
        c_truncated = lsp_end_1based - seq_len
        lsp_end_1based = seq_len
        extra_n = min(c_truncated, lsp_start_1based - 1)
        lsp_start_1based -= extra_n
        c_truncated -= extra_n

    lsp_sequence = protein_sequence[lsp_start_1based - 1 : lsp_end_1based]
    core_epitope = protein_sequence[core_start_1based - 1 : core_end_1based]

    return LSPDesign(
        core_epitope         = core_epitope,
        core_start_1based    = core_start_1based,
        core_end_1based      = core_end_1based,
        lsp_sequence         = lsp_sequence,
        lsp_start_1based     = lsp_start_1based,
        lsp_end_1based       = lsp_end_1based,
        target_length        = lsp_length,
        actual_length        = len(lsp_sequence),
        n_flank_truncated    = n_truncated,
        c_flank_truncated    = c_truncated,
    )


def build_lsp_from_peptide(
    protein_sequence: str,
    core_peptide: str,
    *,
    lsp_length: int = 27,
) -> LSPDesign:
    """Locate ``core_peptide`` in ``protein_sequence`` and build an LSP.

    Raises ValueError if the core does not occur in the protein, or occurs
    more than once (ambiguous — the caller must pass an explicit position).
    """
    occurrences = []
    start = 0
    while True:
        idx = protein_sequence.find(core_peptide, start)
        if idx < 0:
            break
        occurrences.append(idx + 1)   # 1-based
        start = idx + 1
    if not occurrences:
        raise ValueError(f"core peptide {core_peptide!r} not found in protein sequence")
    if len(occurrences) > 1:
        raise ValueError(
            f"core peptide {core_peptide!r} occurs {len(occurrences)} times in "
            f"protein at positions {occurrences}; pass start_1based explicitly"
        )
    return build_lsp(
        protein_sequence,
        core_start_1based=occurrences[0],
        core_length=len(core_peptide),
        lsp_length=lsp_length,
    )
