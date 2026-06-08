"""AIKI-HLAScan: protein-level immunogenicity hot-spot screening on top of AIKI-HLA.

See the design doc in the private repo (`docs/aiki_hlascan_design.md`) for the
full feature matrix and roadmap; this file mirrors that surface.

Pipeline stages:

    Stage 0 (extract):     sliding-window 8-11mers (Class I) / 13-17mers (Class II)
    Stage 1 (alleles):     panel = iedb27 / supertype / custom / patient
    Stage 2 (subsample):   uniform-stride pair-budget fitting (replaces C-terminal chop)
    Stage 3 (score):       handled by host (Modal `/scan` or `aiki_hla.inference`)
    Stage 4 (aggregate):   per-residue MAX-then-COUNT + hot-spots + top-K risky peptides
    Stage 5 (report):      JSON serialization
    Stage 6 (mutations):   per-position 20-AA sweep — deimmunization + neoantigen visibility
    Stage 7 (PHBR):        patient HLA binding rank (Marty et al., Cell 2017)
    Stage 8 (DAI):         differential agretopicity (Duan et al., Nat Genet 2014;
                           Łuksza et al., Nature 2017)
    Stage 9 (LSP):         25–30mer long synthetic peptide construction (BNT122 / Moderna)

Orchestrator hosts:

  - **Modal**: `release/modal/aikihla_app.py` :: `/scan`, `/score_mutations`,
    `/score_neoantigens` POST routes, batched via the warmed AikiMhcModel container.
  - **Local pip CLI**: `aiki-hla scan` drives `aiki_hla.inference.score_csv`
    against the on-the-fly ESM-2 backbone.
"""
from aiki_hla.scan.extract import (
    extract_peptides,
    unique_peptide_strings,
    CLASS_I_LENGTHS,
    CLASS_II_LENGTHS,
    Peptide,
)
from aiki_hla.scan.alleles import (
    resolve_allele_panel,
    AllelePanel,
    PANELS_DIR,
    VALID_MODES,
)
from aiki_hla.scan.aggregate import (
    aggregate_hotspots,
    HotspotResult,
    Hotspot,
    RiskyPeptide,
    DEFAULT_BINDER_MODE,
    DEFAULT_BINDER_PERCENTILE,
    DEFAULT_BINDER_THRESHOLD,
    DEFAULT_HOTSPOT_THRESHOLD,
    DEFAULT_HOTSPOT_PROMISCUITY,
    DEFAULT_HOTSPOT_MIN_RUN,
    DEFAULT_HOTSPOT_MERGE_GAP,
)
from aiki_hla.scan.report import write_report
from aiki_hla.scan.subsample import (
    compute_stride,
    uniform_stride_subsample,
    fit_peptides_to_pair_budget,
)
from aiki_hla.scan.mutations import (
    score_mutations,
    best_deimmunization,
    best_neoantigen_visibility,
    MutationSweepResult,
    SubstitutionResult,
)
from aiki_hla.scan.phbr import (
    compute_phbr,
    is_neoantigen_candidate,
)
from aiki_hla.scan.dai import (
    parse_mutation_spec,
    apply_mutation,
    mutation_spec_to_peptides,
    differential_agretopicity,
    score_neoantigens,
    MutationSpec,
)
from aiki_hla.scan.lsp import (
    build_lsp,
    build_lsp_from_peptide,
    LSPDesign,
)

__all__ = [
    # Stage 0
    "extract_peptides", "unique_peptide_strings",
    "CLASS_I_LENGTHS", "CLASS_II_LENGTHS", "Peptide",
    # Stage 1
    "resolve_allele_panel", "AllelePanel", "PANELS_DIR", "VALID_MODES",
    # Stage 2
    "compute_stride", "uniform_stride_subsample", "fit_peptides_to_pair_budget",
    # Stage 4
    "aggregate_hotspots", "HotspotResult", "Hotspot", "RiskyPeptide",
    "DEFAULT_BINDER_MODE", "DEFAULT_BINDER_PERCENTILE", "DEFAULT_BINDER_THRESHOLD",
    "DEFAULT_HOTSPOT_THRESHOLD", "DEFAULT_HOTSPOT_PROMISCUITY",
    "DEFAULT_HOTSPOT_MIN_RUN", "DEFAULT_HOTSPOT_MERGE_GAP",
    # Stage 5
    "write_report",
    # Stage 6
    "score_mutations", "best_deimmunization", "best_neoantigen_visibility",
    "MutationSweepResult", "SubstitutionResult",
    # Stage 7
    "compute_phbr", "is_neoantigen_candidate",
    # Stage 8
    "parse_mutation_spec", "apply_mutation", "mutation_spec_to_peptides",
    "differential_agretopicity", "score_neoantigens", "MutationSpec",
    # Stage 9
    "build_lsp", "build_lsp_from_peptide", "LSPDesign",
]
