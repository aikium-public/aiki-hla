"""AIKI-HLAScan: protein-level immunogenicity hot-spot screening on top of AIKI-HLA.

Phase 1: pure-Python stages bundled here (extract, allele-panel, aggregate, report).
The orchestrator (peptide × allele scoring) is delegated to the host:

  - **Modal**: `release/modal/aikihla_app.py` :: `/scan` POST route, batched via
    the warmed AikiMhcModel container.
  - **Local pip CLI**: a future `aiki-hla scan` subcommand that drives
    `aiki_hla.inference.score_csv` against the on-the-fly ESM-2 backbone.

Public surface:
    extract_peptides       Stage 0: sliding-window 8-11mers (Class I) / 13-17mers (Class II)
    resolve_allele_panel   Stage 1: panel = iedb27 / supertype / custom
    aggregate_hotspots     Stage 4: per-residue promiscuity + contiguous hot-spots + top-K risky peptides
    write_report           Stage 4: JSON serialization
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

__all__ = [
    "extract_peptides",
    "unique_peptide_strings",
    "CLASS_I_LENGTHS",
    "CLASS_II_LENGTHS",
    "Peptide",
    "resolve_allele_panel",
    "AllelePanel",
    "PANELS_DIR",
    "VALID_MODES",
    "aggregate_hotspots",
    "HotspotResult",
    "Hotspot",
    "RiskyPeptide",
    "DEFAULT_BINDER_MODE",
    "DEFAULT_BINDER_PERCENTILE",
    "DEFAULT_BINDER_THRESHOLD",
    "DEFAULT_HOTSPOT_THRESHOLD",
    "DEFAULT_HOTSPOT_PROMISCUITY",
    "DEFAULT_HOTSPOT_MIN_RUN",
    "DEFAULT_HOTSPOT_MERGE_GAP",
    "write_report",
]
