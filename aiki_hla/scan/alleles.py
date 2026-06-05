"""Stage 1 — allele panel selection.

Six selection modes per docs/immunogenicity_screening_tool_plan.md §5:
    iedb27      — IEDB 27-allele Class I and/or Class II reference panel
    supertype   — Sette HLA-supertype representatives (cheapest, 9 + 5 alleles)
    custom      — user-supplied allele list

Modes deferred to Phase 2: ``topN``, ``region``, ``patient``, ``blast``.

Population-frequency weighting is uniform in Phase 1 (each allele weighted
equally). HLAfreq-data-driven weighting will land in Phase 2 alongside the
``region`` and ``topN`` modes.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


# Allele reference panels are shipped as package data assets (config-like
# reference lists, not gitignored corpus/embedding artifacts). They travel
# with the package so the eventual public-mirror port carries them too.
PANELS_DIR = Path(__file__).resolve().parent / "panels"

VALID_MODES = ("iedb27", "supertype", "custom")


@dataclass(frozen=True)
class AllelePanel:
    """A resolved allele panel ready to drive Stage 3 inference."""
    alleles: tuple[str, ...]
    mhc_class: str          # "I" | "II" | "both"
    mode: str               # one of VALID_MODES
    panel_name: str         # human-readable label
    description: str
    # Per-allele population-frequency weights (uniform = 1.0 in Phase 1).
    # Same order as ``alleles``.
    weights: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if self.weights is not None and len(self.weights) != len(self.alleles):
            raise ValueError(
                f"weights length {len(self.weights)} != alleles length {len(self.alleles)}"
            )
        if self.mhc_class not in ("I", "II", "both"):
            raise ValueError(f"mhc_class must be 'I', 'II', or 'both'; got {self.mhc_class!r}")
        if self.mode not in VALID_MODES:
            raise ValueError(f"mode must be one of {VALID_MODES}; got {self.mode!r}")

    def weight_for(self, allele: str) -> float:
        """Return the population-frequency weight for an allele.

        Returns 1.0 if the panel uses uniform weighting AND the allele is in
        the panel. Returns 0.0 for any allele not in the panel (regardless of
        weighting mode) to make accidental out-of-panel scoring defensible.
        """
        if allele not in self.alleles:
            return 0.0
        if self.weights is None:
            return 1.0
        idx = self.alleles.index(allele)
        return self.weights[idx]


def _load_panel_file(filename: str) -> dict:
    path = PANELS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Allele panel file not found at {path}. "
            f"Expected data/applications/hlascan/{filename} to be checked in."
        )
    with path.open() as f:
        return json.load(f)


def _iedb27(mhc_class: str) -> AllelePanel:
    if mhc_class in ("I", "both"):
        ci = _load_panel_file("iedb27_class_i.json")
    else:
        ci = {"alleles": []}
    if mhc_class in ("II", "both"):
        cii = _load_panel_file("iedb27_class_ii.json")
    else:
        cii = {"alleles": []}

    alleles: list[str] = []
    if mhc_class in ("I", "both"):
        alleles.extend(ci["alleles"])
    if mhc_class in ("II", "both"):
        alleles.extend(cii["alleles"])

    if mhc_class == "I":
        panel_name = "iedb27_class_i"
        description = ci.get("description", "")
    elif mhc_class == "II":
        panel_name = "iedb27_class_ii"
        description = cii.get("description", "")
    else:
        panel_name = "iedb27_both"
        description = (
            "Combined IEDB-27 Class I + Class II reference panels "
            "(54 alleles total)."
        )

    return AllelePanel(
        alleles=tuple(alleles),
        mhc_class=mhc_class,
        mode="iedb27",
        panel_name=panel_name,
        description=description,
    )


def _supertype(mhc_class: str) -> AllelePanel:
    raw = _load_panel_file("supertype_representatives.json")

    alleles: list[str] = []
    if mhc_class in ("I", "both"):
        alleles.extend(raw["class_i"]["alleles"])
    if mhc_class in ("II", "both"):
        alleles.extend(raw["class_ii"]["alleles"])

    if mhc_class == "I":
        panel_name = "supertype_class_i"
    elif mhc_class == "II":
        panel_name = "supertype_class_ii"
    else:
        panel_name = "supertype_both"

    return AllelePanel(
        alleles=tuple(alleles),
        mhc_class=mhc_class,
        mode="supertype",
        panel_name=panel_name,
        description=raw["description"],
    )


def _custom(custom_alleles: list[str], mhc_class: str) -> AllelePanel:
    if not custom_alleles:
        raise ValueError("custom mode requires a non-empty custom_alleles list")
    return AllelePanel(
        alleles=tuple(custom_alleles),
        mhc_class=mhc_class,
        mode="custom",
        panel_name="custom",
        description=f"User-supplied panel of {len(custom_alleles)} allele(s).",
    )


def resolve_allele_panel(
    mode: str = "iedb27",
    mhc_class: str = "both",
    custom_alleles: list[str] | None = None,
) -> AllelePanel:
    """Resolve an allele-selection mode + class into a concrete AllelePanel.

    Args:
        mode: One of ``iedb27``, ``supertype``, ``custom``.
        mhc_class: ``I``, ``II``, or ``both``. Filters the panel by class.
        custom_alleles: When ``mode='custom'``, the list of allele names.

    Returns:
        An ``AllelePanel`` whose ``.alleles`` tuple is the resolved set.
    """
    if mode == "iedb27":
        return _iedb27(mhc_class)
    if mode == "supertype":
        return _supertype(mhc_class)
    if mode == "custom":
        if not custom_alleles:
            raise ValueError("custom mode requires --custom-alleles")
        return _custom(custom_alleles, mhc_class)
    raise ValueError(
        f"Unknown allele mode {mode!r}. Phase 1 supports: {VALID_MODES}. "
        f"Phase 2 will add: topN, region, patient, blast."
    )
