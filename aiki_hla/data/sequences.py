"""MHC G-domain sequence lookup.

Resolves an HLA allele name (e.g. ``"HLA-A*02:01"``, ``"HLA-DRB1*01:01"``,
``"HLA-DQA1*05:01-DQB1*03:01"`` for Class II heterodimers) to its 182-residue
G-domain protein sequence.

The bundled database (``mhc_sequences.json``) covers every allele in the
released training corpus (n=400 in v1.0). For alleles outside this set,
the canonical 37,480-entry database ships separately on Zenodo; see the
`extend_database` helper below.

The G-domain construction follows the manuscript Methods §"Model, training,
and ensemble":

    Class I:   α1 (residues 1-91) + α2 (residues 92-182)
    Class II:  α1 (91 residues) + β1 (91 residues), concatenated end-to-end

Both classes produce a uniform 182-residue sequence; this is the input
length the deployed model expects.

Public API:
    get_g_domain_sequence(allele)  -> str | None
    list_alleles()                 -> list[str]
    is_supported(allele)           -> bool
    extend_database(json_path)     -> None  (in-memory; not persisted)
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

_DATA_PATH = Path(__file__).resolve().parent / "mhc_sequences.json"


@lru_cache(maxsize=1)
def _load_db() -> dict:
    """Lazy-load the bundled allele database (cached)."""
    with _DATA_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_g_domain_sequence(allele: str) -> str | None:
    """Return the 182-residue G-domain sequence for `allele`, or None if unknown.

    Allele names use the standard HLA nomenclature:
        Class I:   "HLA-A*02:01", "HLA-B*07:02", "HLA-C*07:02", ...
        Class II:  "HLA-DRB1*01:01" (β-only legacy form), or
                   "HLA-DQA1*05:01-DQB1*03:01" (α-β heterodimer; preferred)
    """
    db = _load_db()
    entry = db["alleles"].get(allele)
    return entry["g_domain_sequence"] if entry else None


def get_mhc_class(allele: str) -> str | None:
    """Return 'I' or 'II' for a supported allele; None if unknown."""
    db = _load_db()
    entry = db["alleles"].get(allele)
    return entry["mhc_class"] if entry else None


def list_alleles() -> list[str]:
    """All HLA alleles supported out-of-the-box."""
    return sorted(_load_db()["alleles"].keys())


def is_supported(allele: str) -> bool:
    """Whether the bundled DB has a G-domain sequence for `allele`."""
    return allele in _load_db()["alleles"]


def extend_database(json_path: str | Path) -> int:
    """Merge an additional JSON DB into the in-memory cache.

    Useful when the user has downloaded the full 37,480-entry Zenodo
    database. The original bundled DB is not modified on disk.

    Returns the number of new alleles added.
    """
    with Path(json_path).open("r", encoding="utf-8") as f:
        extra = json.load(f)
    db = _load_db()
    before = len(db["alleles"])
    for k, v in extra.get("alleles", {}).items():
        db["alleles"].setdefault(k, v)
    return len(db["alleles"]) - before
