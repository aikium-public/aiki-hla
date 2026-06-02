"""AIKI-HLA: a calibrated peptide-MHC binding predictor.

Public-facing API. Most imports are lazy to keep ``import aiki_hla``
fast and to let the package install without torch present (for
environments that only want the manifest/data tools or the gate-only
viability score).

Two regimes are exposed:

  - **Binding** — pure AIKI per-(peptide, allele) score. Use for
    allele-specific cluster-novel discrimination (manuscript Table 1
    lower bound 0.765, upper bound 0.913).

    >>> from aiki_hla import score
    >>> p = score(peptide="GILGFVFTL", allele="HLA-A*02:01")

  - **Presentation** — composite ``sqrt(p_ligand * p_binding)`` with the
    peptide-only ligand-likeness viability gate (manuscript Methods
    §"Viability gate construction and composition"). Use for off-target
    safety screening, neoantigen ranking, and proteome-wide scoring;
    the cluster-novel composite hits 0.816 on the 39-allele strict
    stratum (+0.060 over pure AIKI) and reduces off-target FPR by
    20–280×. The Python API returns the manuscript's ``p_ligand`` as
    ``p_gate`` and ``p_binding`` as ``p_aiki`` for programmatic
    consistency with the function names.

    >>> from aiki_hla import score_composite
    >>> out = score_composite(peptide="GILGFVFTL", allele="HLA-A*02:01",
    ...                       mode="both")
    >>> out["p_aiki"], out["p_gate"], out["p_composite"]

  - **Gate-only** — the viability gate alone (no MHC), as a cheap
    pre-filter:

    >>> from aiki_hla import score_gate
    >>> g = score_gate("GILGFVFTL", mhc_class="I")
    >>> g.p_gate, g.quality

The default of :func:`score` is ``mode="binding"`` for backward
compatibility with v1.0.0 callers; pick the composite explicitly when
the regime calls for it. See ``aiki_hla.viability_gate`` for the
mechanism (negative-class mismatch with the field, hand-crafted
features, equal-weight geometric-mean composition with no fitted
parameter).

Reproducibility:
    python -m validation.reproduce_paper_numbers --deposit <zenodo-doi>
"""
__version__ = "1.0.0"

# Names served by lazy attribute access. Grouped by sub-module so the
# heavy torch import in `inference` only fires when binding is actually
# called — gate-only callers stay torch-free.
_INFERENCE_NAMES = {
    "score", "score_csv", "score_dataframe",
    "load_trained_model", "ensure_embeddings",
}
_GATE_NAMES = {
    "score_gate", "score_gate_batch", "score_composite",
    "compose", "GateScore",
    "CI_LENGTH_RANGE", "CII_LENGTH_RANGE",
}


def __getattr__(name):
    if name in _INFERENCE_NAMES:
        from aiki_hla import inference as _inf
        return getattr(_inf, name)
    if name in _GATE_NAMES:
        from aiki_hla import viability_gate as _vg
        return getattr(_vg, name)
    raise AttributeError(f"module 'aiki_hla' has no attribute {name!r}")


__all__ = [
    "__version__",
    # binding (AIKI-HLA per-(peptide, allele) score)
    "score",
    "score_csv",
    "score_dataframe",
    "load_trained_model",
    "ensure_embeddings",
    # presentation (AIKI × viability gate composite)
    "score_composite",
    "compose",
    # gate-only (peptide-only viability score, no MHC)
    "score_gate",
    "score_gate_batch",
    "GateScore",
    "CI_LENGTH_RANGE",
    "CII_LENGTH_RANGE",
]
