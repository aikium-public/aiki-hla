"""Peptide-only MHC-epitope viability gate (Class I and Class II).

The viability gate is the second model AIKI-HLA ships. It scores
"presentation likeness" — how plausible a peptide is as *any* HLA ligand
— from hand-crafted features (no learned embedding). Two class-specific
gates are shipped:

  - Class I  — held-out AUROC 0.869 (0.900 vs pure proteome decoys);
    trained on 8–11-mers.
  - Class II — held-out AUROC 0.718 (0.813 vs pure proteome decoys);
    trained on 12–25-mers.

It exists because of the negative-class mismatch between AIKI-HLA and
the field: AIKI trains on *cross-allele real-peptide* hard negatives
(learns allele-specificity), but proteome screening regimes ask "is
this random sequence a ligand at all?" — for which AIKI lacks
background-rejection training signal. The gate fills that gap with
~zero additional inference cost.

Composition with AIKI-HLA is the **equal-weight geometric mean**, fixed
per the manuscript Methods §"Viability gate construction and
composition":

    p_composite = sqrt(p_ligand * p_binding)         (eps-guarded)
                = sqrt(p_gate   * p_aiki   )         (API field names)

The manuscript notation (``p_ligand`` for the gate's ligand-likeness
score, ``p_binding`` for AIKI's binding probability) is what appears in
the paper; the Python API names (``p_gate``, ``p_aiki``) are what the
return dicts use, for programmatic consistency with the function
names. Same numbers, two names.

Use cases:

  - **`score(peptide, allele)`** — pure AIKI; allele-specific
    cluster-novel discrimination (manuscript Table 1 lower bound 0.765,
    upper bound 0.913). This is the default for backward compatibility
    with v1.0.0 callers.
  - **`score_composite(peptide, allele, mode="presentation")`** — the
    composed score; recommended for off-target safety screening,
    neoantigen ranking against random background, and proteome-wide
    flagging (cluster-novel composite 0.816 on the 39-allele strict
    stratum, +0.060 over pure AIKI; 20–280× off-target FPR reduction).
  - **`score_gate(peptide, mhc_class)`** — gate only; useful as a
    cheap pre-filter that needs no MHC sequence.

Note: this is the *inference* viability gate. There is also a
**build-time contamination audit gate** at ``aiki_hla.data.audit_gate``
that runs at training entry; the two have unrelated purposes despite
sharing the word "gate". Keep them straight.
"""
from __future__ import annotations

import logging
import pickle
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

logger = logging.getLogger(__name__)

# ── Feature definitions (must match the trained gates exactly) ──────────
# The 148-d feature vector is: 20-d amino-acid composition (frequency) +
# 8 scalar properties (length, hydropathy mean/std, net & abs charge,
# %hydrophobic / %acidic / %basic) + 6×20 terminal-residue one-hots
# (positions 1, 2, 3, n-2, n-1, n — anchor-capturing, length-agnostic).
# DO NOT change the ordering, AA index, KD scale, or terminal positions
# — the pickled sklearn classifiers were fit on this exact layout.

_AA = "ACDEFGHIKLMNPQRSTVWY"
_AIDX = {a: i for i, a in enumerate(_AA)}
_AASET = set(_AA)
# Kyte–Doolittle hydropathy (matches build_viability_gate.py:25).
_KD = {
    'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
    'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
    'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
    'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2,
}
# Net charge (matches build_viability_gate.py:27).
_CHG = {'D': -1.0, 'E': -1.0, 'K': 1.0, 'R': 1.0, 'H': 0.1}

# Gate training-length ranges (matches build_class("I", 8, 11, …) and
# build_class("II", 12, 25, …) in build_viability_gate.py:137).
CI_LENGTH_RANGE = (8, 11)
CII_LENGTH_RANGE = (12, 25)

# Feature vector dimensionality. Asserted at gate-load time.
_FEATURE_DIM = 148

# Package-data location for the trained gate pickles.
_DATA_DIR = Path(__file__).resolve().parent / "data"
_GATE_FILES = {
    "I":  _DATA_DIR / "viability_gate_CI.pkl",
    "II": _DATA_DIR / "viability_gate_CII.pkl",
}

# Composition epsilon — matches compose_gate_aiki.py:62.
_COMPOSE_EPS = 1e-6


@dataclass(frozen=True)
class GateScore:
    """One peptide's gate score plus a length-in-range quality flag.

    Attributes:
        p_gate: GradientBoosting gate probability in [0, 1].
        mhc_class: "I" or "II"; the gate used.
        quality: "ok" (length in gate's training range) or
            "out_of_range" (gate extrapolated — interpret with caution).
    """
    p_gate: float
    mhc_class: str
    quality: str


def _featurise_one(peptide: str) -> np.ndarray:
    """Compute the 148-d feature vector for a single peptide.

    Mirrors ``feats(p)`` in build_viability_gate.py:32 byte-for-byte.

    Raises:
        ValueError: if the peptide contains a non-canonical amino acid.
    """
    p = peptide.upper().strip()
    n = len(p)
    if n == 0:
        raise ValueError("peptide is empty")
    bad = set(p) - _AASET
    if bad:
        raise ValueError(
            f"peptide contains non-canonical amino acid(s) {sorted(bad)!r}; "
            f"the gate's feature function is defined only over {_AA!r}"
        )
    comp = np.zeros(20)
    for c in p:
        comp[_AIDX[c]] += 1
    comp /= n
    hyd = np.array([_KD[c] for c in p])
    chg = np.array([_CHG.get(c, 0.0) for c in p])
    base = [
        n, hyd.mean(), hyd.std(), chg.sum(), np.abs(chg).sum(),
        sum(c in "FILMVWY" for c in p) / n,   # %hydrophobic
        sum(c in "DE" for c in p) / n,        # %acidic
        sum(c in "KR" for c in p) / n,        # %basic
    ]
    term = np.zeros(6 * 20)
    pos = [0, 1, 2, n - 3, n - 2, n - 1]
    for j, pp in enumerate(pos):
        if 0 <= pp < n:
            term[j * 20 + _AIDX[p[pp]]] = 1
    return np.concatenate([comp, base, term])


def _featurise_many(peptides: Iterable[str]) -> np.ndarray:
    return np.vstack([_featurise_one(p) for p in peptides])


# Lazy singletons — load each gate at most once per process.
_GATES_CACHE: dict[str, object] = {}


def _load_gate(mhc_class: str):
    """Load and cache a class-specific gate's GradientBoosting classifier."""
    mhc_class = mhc_class.upper()
    if mhc_class not in _GATE_FILES:
        raise ValueError(f"mhc_class must be 'I' or 'II'; got {mhc_class!r}")
    if mhc_class in _GATES_CACHE:
        return _GATES_CACHE[mhc_class]
    path = _GATE_FILES[mhc_class]
    if not path.exists():
        raise FileNotFoundError(
            f"viability gate pickle missing at {path}. The gate is shipped as "
            f"package data; reinstall the aiki-hla package to recover it."
        )
    with path.open("rb") as f:
        bundle = pickle.load(f)
    if "gb" not in bundle:
        raise RuntimeError(
            f"viability gate pickle {path} has unexpected keys "
            f"{sorted(bundle.keys())}; expected {{'lr', 'gb'}}."
        )
    clf = bundle["gb"]
    n_in = getattr(clf, "n_features_in_", None)
    if n_in is not None and n_in != _FEATURE_DIM:
        raise RuntimeError(
            f"viability gate pickle {path} expects {n_in}-d features; "
            f"this package's _featurise_one returns {_FEATURE_DIM}-d. "
            f"Gate / feature-function are out of sync — package install corrupt."
        )
    _GATES_CACHE[mhc_class] = clf
    return clf


def _infer_class_from_allele(allele: str) -> str:
    """Heuristic: DRB/DPA/DPB/DQA/DQB/DOA/DOB/DMA/DMB → Class II; else I.

    Matches the convention used elsewhere in the package
    (``src/v2/applications/hlascan/scan.py:_infer_allele_class``).
    """
    a = allele.upper()
    if any(tag in a for tag in ("DRB", "DPA", "DPB", "DQA", "DQB", "DOA", "DOB", "DMA", "DMB")):
        return "II"
    return "I"


def _infer_class_from_length(peptide: str) -> str:
    """Length-based class fallback when no allele is given.

    Class I ligands cluster around 9.9 aa, Class II around 15.7 aa. The
    boundary at 12 matches the training-range split in
    ``build_viability_gate.py``: CI 8–11, CII 12–25.
    """
    return "I" if len(peptide) <= 11 else "II"


def _check_length_quality(peptide: str, mhc_class: str) -> str:
    """Return ``"ok"`` if the peptide length is in the gate's training
    range, ``"out_of_range"`` otherwise. The gate still runs (no
    exception, no silent skip — per the project's "no silent fallbacks"
    rule); the caller decides how to use the score.
    """
    n = len(peptide)
    lo, hi = (CI_LENGTH_RANGE if mhc_class == "I" else CII_LENGTH_RANGE)
    if lo <= n <= hi:
        return "ok"
    warnings.warn(
        f"peptide length {n} is outside the Class {mhc_class} gate's "
        f"training range [{lo}, {hi}]; gate will extrapolate "
        f"(returned quality='out_of_range').",
        category=UserWarning,
        stacklevel=3,
    )
    return "out_of_range"


def score_gate(
    peptide: str,
    *,
    mhc_class: str | None = None,
    allele: str | None = None,
) -> GateScore:
    """Score one peptide with the viability gate.

    The gate is peptide-only — it does not consume the MHC sequence. The
    ``mhc_class`` / ``allele`` arguments only select which of the two
    class-specific gates to use.

    Args:
        peptide: Amino-acid sequence, canonical 20 AAs.
        mhc_class: ``"I"`` or ``"II"``. If omitted, inferred from
            ``allele`` (preferred) or from ``len(peptide)``.
        allele: HLA allele name; used to infer the class when
            ``mhc_class`` is omitted.

    Returns:
        ``GateScore(p_gate, mhc_class, quality)``. ``quality`` is
        ``"ok"`` if the peptide length is in the gate's training range
        and ``"out_of_range"`` otherwise (in which case the gate's
        prediction is an extrapolation; a warning is also emitted).

    Raises:
        ValueError: on a non-canonical amino acid or an unresolvable class.
    """
    if mhc_class is None:
        mhc_class = (
            _infer_class_from_allele(allele) if allele
            else _infer_class_from_length(peptide)
        )
    mhc_class = mhc_class.upper()
    if mhc_class not in ("I", "II"):
        raise ValueError(f"mhc_class must be 'I' or 'II'; got {mhc_class!r}")
    clf = _load_gate(mhc_class)
    x = _featurise_one(peptide).reshape(1, -1)
    p = float(clf.predict_proba(x)[0, 1])
    quality = _check_length_quality(peptide, mhc_class)
    return GateScore(p_gate=p, mhc_class=mhc_class, quality=quality)


def score_gate_batch(
    peptides: list[str],
    *,
    mhc_class: str | None = None,
    alleles: list[str] | None = None,
) -> list[GateScore]:
    """Vectorised batch version of :func:`score_gate`.

    All peptides are scored on the same class if ``mhc_class`` is
    given; otherwise the class is inferred per-peptide (from
    ``alleles[i]`` if supplied, else from peptide length). Peptides
    whose class differs are silently routed to the matching gate;
    feature vectors are batched within each class for speed.
    """
    if alleles is not None and len(alleles) != len(peptides):
        raise ValueError(
            f"alleles list length ({len(alleles)}) must equal peptides "
            f"length ({len(peptides)})"
        )
    # Resolve class per row.
    classes: list[str] = []
    for i, pep in enumerate(peptides):
        if mhc_class is not None:
            cls = mhc_class.upper()
        elif alleles is not None:
            cls = _infer_class_from_allele(alleles[i])
        else:
            cls = _infer_class_from_length(pep)
        if cls not in ("I", "II"):
            raise ValueError(f"row {i}: cannot resolve class to 'I' or 'II'")
        classes.append(cls)
    # Batch per class.
    out: list[GateScore | None] = [None] * len(peptides)
    for cls in ("I", "II"):
        idx = [i for i, c in enumerate(classes) if c == cls]
        if not idx:
            continue
        clf = _load_gate(cls)
        X = _featurise_many([peptides[i] for i in idx])
        probs = clf.predict_proba(X)[:, 1]
        for j, i in enumerate(idx):
            q = _check_length_quality(peptides[i], cls)
            out[i] = GateScore(p_gate=float(probs[j]), mhc_class=cls, quality=q)
    return [o for o in out if o is not None]  # type: ignore[misc]


def compose(p_aiki: float, p_gate: float) -> float:
    """Equal-weight geometric mean: ``sqrt(p_binding * p_ligand)``.

    In manuscript notation this is ``sqrt(p_binding * p_ligand)``; in the
    Python API the arguments are named ``p_aiki`` and ``p_gate``
    respectively (same numbers, names chosen for programmatic
    consistency with the function names).

    Eps-guarded (``eps = 1e-6``) to keep the gradient defined and to
    match the composition used in the manuscript and in
    ``scripts/applications/compose_gate_aiki.py``.

    The geometric mean was chosen over arithmetic in the manuscript
    because it forces both inputs to be confident: a near-zero in
    either factor pulls the composite low, which is the desired
    behaviour for screening (one strong rejection beats one strong
    acceptance).
    """
    pa = float(np.clip(p_aiki, _COMPOSE_EPS, 1.0))
    pg = float(np.clip(p_gate, _COMPOSE_EPS, 1.0))
    return float(np.sqrt(pa * pg))


def score_composite(
    peptide: str,
    allele: str,
    *,
    mode: str = "both",
    device: str = "cuda",
) -> dict:
    """Score ``peptide × allele`` with both AIKI-HLA and the gate.

    This is the **regime-aware composite** discussed in the
    manuscript's abstract and Methods §"Viability gate construction
    and composition":

        p_composite = sqrt(p_aiki * p_gate)

    Args:
        peptide: Amino-acid sequence.
        allele: HLA allele name (e.g. ``"HLA-A*02:01"``).
        mode: ``"binding"`` (pure AIKI; back-compat with v1.0.0),
            ``"presentation"`` (composite only), or ``"both"`` (both
            scores returned; this is the default for this entry point
            because the caller has explicitly asked for the composite
            function — see :func:`aiki_hla.score` for the
            mode=``"binding"`` default of the canonical entry point).
        device: Torch device for the AIKI forward pass.

    Returns:
        ``{"p_aiki": float, "p_gate": float, "p_composite": float,
        "mhc_class": "I"|"II", "quality": "ok"|"out_of_range"}``.
        ``p_aiki`` is ``None`` if ``mode == "presentation"`` and the
        caller did not also request binding; ``p_composite`` is
        ``None`` if ``mode == "binding"``.
    """
    from aiki_hla.inference import score as _aiki_score

    if mode not in ("binding", "presentation", "both"):
        raise ValueError(
            f"mode must be one of 'binding', 'presentation', 'both'; "
            f"got {mode!r}"
        )
    cls = _infer_class_from_allele(allele)
    gate = score_gate(peptide, mhc_class=cls)
    p_aiki: float | None = None
    p_composite: float | None = None
    if mode in ("binding", "both"):
        p_aiki = float(_aiki_score(peptide=peptide, allele=allele, device=device))
    if mode in ("presentation", "both"):
        if p_aiki is None:
            p_aiki_local = float(_aiki_score(peptide=peptide, allele=allele, device=device))
        else:
            p_aiki_local = p_aiki
        p_composite = compose(p_aiki_local, gate.p_gate)
    return {
        "p_aiki": p_aiki,
        "p_gate": gate.p_gate,
        "p_composite": p_composite,
        "mhc_class": gate.mhc_class,
        "quality": gate.quality,
    }


__all__ = [
    "CI_LENGTH_RANGE",
    "CII_LENGTH_RANGE",
    "GateScore",
    "compose",
    "score_gate",
    "score_gate_batch",
    "score_composite",
]
