# Changelog

## 1.1.0 — Cancer-vaccine application surface + manuscript reframe (2026-06-07; documentation update 2026-06-17)

The reference implementation gains the cancer-vaccine half of the
AIKI-HLAScan surface, and the companion manuscript is reframed
around the *measurement* question (see also Zenodo v1.1.0 at
[10.5281/zenodo.20583336](https://doi.org/10.5281/zenodo.20583336),
concept [10.5281/zenodo.20520819](https://doi.org/10.5281/zenodo.20520819)).

Now accompanies *Measuring peptide–MHC generalization to unseen
alleles across both HLA classes* (Mysore 2026, submitted to PLOS
Computational Biology). The 4.66M / 5.8M open / research-use
ensembles, the leakage-controlled benchmark, and the per-allele
median AUROC headlines (0.913 conventional split, 0.765 on 39
strictly-novel alleles, 0.949 on 111 rare subtypes) are unchanged.

### New in `aiki_hla.scan`

- `aiki_hla.scan.subsample` — uniform-stride pair-budget fitting
  (replaces the prior C-terminal chop that silently dropped the back
  of large proteins).
- `aiki_hla.scan.mutations` — per-position 20-AA sweep, callable-agnostic
  (caller passes `score_fn`); returns deimmunization and
  neoantigen-visibility views.
- `aiki_hla.scan.phbr` — Patient HLA Binding Rank (Marty *Cell* 2017)
  with per-allele rank percentile, PHBR allele, per-peptide counts
  at the conventional 0.5 % and 2 % rank thresholds.
- `aiki_hla.scan.dai` — Differential Agretopicity Index (Duan *Nat
  Genet* 2014; Łuksza *Nature* 2017): `parse_mutation_spec` accepts
  `G12D` and `12:G>D` forms; `differential_agretopicity` has `log2`
  (Łuksza form) and `diff` (Duan legacy form) modes;
  `score_neoantigens` is the end-to-end DAI pipeline.
- `aiki_hla.scan.lsp` — BioNTech BNT122 / Moderna mRNA-4157 27-mer
  Long Synthetic Peptide cassette with symmetric flanks, overflow
  recovered on the opposite end before truncation.

All scoring continues to route through the AIKI-HLA 3-seed ensemble;
no new ML. The new modules wrap field-standard formulas around
established peptide–MHC scores.

The Modal endpoint surface gains `/score_neoantigens` (DAI-ranked
neoantigen output for a sequence × mutations × patient-HLA input).
Live demo at https://aikium--aikihla-landing-page.modal.run.

## 1.0.0 — Initial release

Open reference implementation accompanying *Aiki-HLA: a Class I and II
peptide–MHC predictor separating ligand-likeness from allele
specificity, with a leakage-controlled benchmark* (Mysore 2026,
Nature Machine Intelligence — subsequently retitled and resubmitted
to PLOS Computational Biology as *Measuring peptide–MHC generalization
to unseen alleles across both HLA classes*; see v1.1.0 above).

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
