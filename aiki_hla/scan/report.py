"""Stage 4 — JSON serialization of the scan result.

Phase 1: JSON only. Phase 2 will add HTML / CSV / PDF outputs per the
design plan §8.6.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from aiki_hla.scan.aggregate import HotspotResult


def to_dict(result: HotspotResult, *, top_k: int = 50) -> dict[str, Any]:
    """Convert a HotspotResult to a serialisable dict.

    Args:
        result: The HotspotResult from ``aggregate_hotspots``.
        top_k: Cap on the number of ``risky_peptides`` to include.

    Returns:
        Dict ready for ``json.dumps``.
    """
    return {
        "schema_version": "aiki-hlascan-1.0",
        "protein_length": result.protein_length,
        "n_peptides_extracted": result.n_peptides_extracted,
        "n_scored_calls": result.n_scored_calls,
        "allele_panel_name": result.allele_panel_name,
        "binder_selection": {
            "mode": result.binder_mode,
            "binder_percentile": result.binder_percentile,
            "binder_threshold": result.binder_threshold,
        },
        "thresholds": {
            "binder_threshold": result.binder_threshold,
            "hotspot_threshold": result.hotspot_threshold,
        },
        "aggregate_risk": result.aggregate_risk,
        "per_residue_score": [round(float(s), 4) for s in result.per_residue_score],
        "hotspots": [
            {
                "start": h.start,
                "end": h.end,
                "length": h.length,
                "max_score": round(h.max_score, 4),
                "mean_score": round(h.mean_score, 4),
                "contributing_alleles": list(h.contributing_alleles),
                "peptides": list(h.peptides),
            }
            for h in result.hotspots
        ],
        "risky_peptides_top_k": [
            {
                "peptide": r.peptide,
                "allele": r.allele,
                "start": r.start,
                "end": r.end,
                "length": r.length,
                "mhc_class": r.mhc_class,
                "binding_prob": round(r.binding_prob, 4),
                "population_weighted_prob": round(r.population_weighted_prob, 4),
                "flags": list(r.flags),
            }
            for r in result.risky_peptides[:top_k]
        ],
        "risky_peptides_count": len(result.risky_peptides),
        "per_allele_n_risky": dict(result.per_allele_n_risky),
    }


def write_report(
    result: HotspotResult,
    out_path: str | Path,
    *,
    top_k: int = 50,
    indent: int = 2,
) -> Path:
    """Write the scan result to ``out_path`` as JSON."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump(to_dict(result, top_k=top_k), f, indent=indent)
    return out
