# Changelog

## 1.0.0 — Initial release

Open reference implementation accompanying *Aiki-HLA: a Class I and II
peptide–MHC predictor separating ligand-likeness from allele
specificity, with a leakage-controlled benchmark* (Mysore 2026,
Nature Machine Intelligence).

### Model

- Class-agnostic single-head architecture: frozen ESM-2 650M backbone
  with a 3.9M-parameter dropout self-attention head over the full
  182-residue MHC G-domain (Class I α1+α2; Class II α1+β1 padded to
  the same length).
- Deployed as a 3-seed ensemble of calibrated probabilities; checkpoint
  identities SHA-256-pinned in the Zenodo deposit's `MANIFEST.json`.
- The released model is the parameter-free geometric mean
  `√(p_ligand · p_binding)` of an allele-blind ligand-likeness score
  (`p_gate`) and an allele-specific binding score (`p_aiki`); both
  scorers ship as package data.

### Public API

- `aiki_hla.score(peptide, allele)` — allele-specific binding
  probability `p_binding` from the 3-seed ensemble.
- `aiki_hla.score_composite(peptide, allele, mode="both")` — combined
  presentation score; returns `p_aiki`, `p_gate`, `p_composite`,
  `mhc_class`, `quality`.
- `aiki_hla.score_gate(peptide, mhc_class=...)` — allele-blind
  ligand-likeness score from hand-crafted features (no MHC sequence,
  no torch required).
- `aiki_hla.score_csv` / `aiki_hla.score_dataframe` — batch scoring.
- `aiki_hla.ensure_embeddings`, `aiki_hla.load_trained_model` — lower-level
  primitives.
- `aiki_hla.compose(p_aiki, p_gate)` — equal-weight geometric-mean
  composition (eps-guarded).

### Command-line

- `aiki-hla score --peptides X --alleles Y --out P --mode {binding,
  presentation, both}` — batch scoring; `--mode` default is `binding`
  (pure `p_binding`); `presentation` returns the combined score in
  place of the AIKI column; `both` adds `p_aiki`, `p_gate`,
  `p_composite`, `gate_quality` columns.
- `aiki-hla gate --peptide P --mhc-class {I, II}` — peptide-only
  ligand-likeness score.
- `aiki-hla lookup --peptide P --allele A [--mode ...]` —
  single-pair lookup.
- `aiki-hla evaluate --predictions P --gold G --strata ...` —
  tiered AUROC + bootstrap CIs.

### Licenses

- Code: Apache-2.0.
- Open-ensemble training corpus and result JSONs: CC-BY-4.0.
- Open-ensemble model weights: CC-BY-4.0.
- Research-use ensemble: model weights CC-BY-4.0; training-corpus
  manifest CC-BY-4.0, but the upstream sources it points at carry
  CC-BY-NC or academic-only licenses — the rebuild script honours each
  upstream license.
