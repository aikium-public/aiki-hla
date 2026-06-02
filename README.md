# AIKI-HLA

A factorized peptide–MHC predictor and a leakage-controlled benchmark
for generalization to novel HLAs.

Under the released benchmark's pair-symmetric cross-tool protocol,
AIKI-HLA **leads eight published open predictor variants** (MHCflurry
2.0, TransPHLA, BigMHC EL, MixMHC2pred v2.0, NetMHCpan-4.1 EL/BA,
NetMHCIIpan-4.1 EL/BA) by per-allele ΔAUROC = **+0.22 to +0.37**
(Wilcoxon p < 10⁻⁹ on every comparator), reports the **first
per-allele AUROC measurement on a pan-class (Class I + Class II)
strictly cluster- and sequence-disjoint stratum** (0.816 composed
score), reaches **0.83 to 0.89 top-100 precision** on Class I alleles
(n_a = 40 to 43 with ≥100 paired rows after the pair-symmetric
filter) versus 0.44 to 0.61 for MHCflurry, TransPHLA, and BigMHC EL
on the same protocol, and **covers 421 (open) / 507 (research-use)
HLA alleles spanning both classes**, well beyond MHCflurry-2.0's
171-allele Class I list and MixMHC2pred-v2.0's 88-allele Class II
list (the next-broadest per-class published allele lists among open
tools we surveyed).

AIKI-HLA scores ligand-likeness and allele-specific binding as two
questions and combines them as the parameter-free geometric mean
`√(p_ligand · p_binding)`, with no fitted mixing parameter. The
allele-specific factor reads the full 182-residue MHC G-domain
through one head shared across both classes (Class I α1+α2; Class II
α1+β1, padded to the same 182-residue length), not the 34-residue
pseudosequence and class-specific models prior tools use. Deployed
as a 3-seed ensemble of calibrated probabilities.

Two ensembles of **identical architecture and recipe** are released,
differing only in the licensing of their training sources:

- **Open ensemble** — trained on a **fully-redistributable** subset
  of the corpus: **4.66 million experimental measurements over 421
  classical HLA alleles** (HLA-A/B/C, HLA-DR, HLA-DP, HLA-DQ; CC-BY-4.0
  / GPL-3.0 / MIT / CC-BY-ND-4.0 sources). No commercial-use
  restriction. Species and class scope match every open comparator,
  making this subset the cross-tool benchmark anchor.
- **Research-use ensemble** — extends the open corpus with 1.06M
  further measurements from CC-BY-NC sources (MHC Motif Atlas,
  SysteMHC Atlas) and an academic-only source (MHCnuggets), plus
  529,129 (peptide, predicted-allele) pairs from model-based
  deconvolution of multi-allelic mass-spectrometry data, plus
  non-classical HLA (E, F, G; MICA, MICB): **5.8 million experimental
  measurements over 507 alleles**. License mix forces a
  **research-only** release.

The 5.8M training corpus is **decoy-free** — only experimental
measurements plus source-matched cross-allele/class negatives, not
random proteome fragments or anchor-shuffled decoys. This is the
largest open, decoy-free pMHC training set among those we surveyed.

Per-allele median AUROC lies in the bracket [0.71, 0.92] depending on
how cluster-novel the query is. The **allele-specific factor**
`p_binding` reaches:
- **0.765** [0.691, 0.797] on a 39-allele strictly-novel sub-stratum
  (lower bound; unchanged between the two ensembles — the
  generalization frontier is a property of the corpus's allele
  distribution, not its volume).
- **0.913** [0.904, 0.924] on the open ensemble's 1% held-out
  validation partition (174 alleles, interpolation upper bound).
- **0.911** [0.893, 0.927] on the research-use ensemble's broader 1%
  held-out (285 alleles — statistically equivalent ceiling on 64% more
  alleles than the open subset).
- **0.949** on the 111 newly-evaluable rare-subtype alleles only the
  research-use ensemble can reach.

On the symmetric leakage-controlled cross-tool protocol scored on the
open ensemble's 1% held-out partition (every test row pair-filtered
against each comparator's training data), the allele-specific factor
exceeds eight published open predictor variants per-allele by
**ΔAUROC = +0.22 to +0.37** (Wilcoxon p < 10⁻⁹ on all eight). Four
fully-redistributable open comparators (MHCflurry 2.0, TransPHLA,
BigMHC EL, MixMHC2pred v2.0) are scored using their published
weights; NetMHCpan-4.1 EL/BA and NetMHCIIpan-4.1 EL/BA are
additionally included via the IEDB Tools API.

The **combined score** `√(p_ligand · p_binding)`, the released model,
raises the strictly-novel score to **0.816** (+0.060 over pure
allele-specific), additionally **equals MHCflurry-2.0 on recent
2024–2026 IEDB depositions held out from all models** (ΔAUROC =
−0.006, n = 48 Class I alleles), and cuts off-target
proteome-screening FPR by **20–280×**.

This repository is the open reference implementation for the manuscript:

> Mysore, V. *Aiki-HLA: a factorized peptide–MHC predictor and a
> leakage-controlled benchmark for generalization to novel HLAs.* (2026).

The companion Zenodo deposit
([10.5281/zenodo.20520819](https://doi.org/10.5281/zenodo.20520819) — concept
DOI, resolves to the latest version; this v1.0.0 release at
[10.5281/zenodo.20520820](https://doi.org/10.5281/zenodo.20520820))
contains the training corpus, splits, checkpoints, and result JSONs that back
every numerical claim.

## Install

```bash
pip install aiki-hla
```

## The released model is the combined score

Per the manuscript: *"AIKI-HLA scores the two [questions] separately
and combines them as the geometric mean `√(p_ligand · p_binding)`,
with no fitted mixing parameter. The combined score is benchmarked
for practical use (off-target safety, neoantigen ranking, proteome
screening); `p_binding` is reported in isolation only where the task
is itself allele-specific (cross-tool comparison, cluster-novel
discrimination)."*

Decision table — pick the function whose semantics match your task:

| Task | Function | Reference number |
|---|---|---|
| Off-target safety / proteome screening / neoantigen ranking against random background (most users) | **`score_composite(...)`** (the released combined score) | strictly-novel 0.816; 20–280× off-target FPR reduction; equals MHCflurry-2.0 on recent 2024–2026 IEDB depositions held out from all models |
| Cross-tool benchmark replication, cluster-novel discrimination (allele-specific tasks) | **`score(peptide, allele)`** (pure `p_binding`) | PA-med 0.765 strictly-novel lower bound; 0.913 upper bound; Δ +0.22 to +0.37 over eight open predictor variants on the pair-symmetric protocol |
| Per-patient neoantigen ranking *within a fixed patient HLA set* | **`score(peptide, allele)`** (no random-background term) | use the allele-specific factor; the allele-blind gate adds noise here, not signal |
| Cheap library triage / pre-filter (no MHC needed) | **`score_gate(peptide, mhc_class)`** | gate held-out AUROC: CI 0.869, CII 0.718 |

The Python API exposes the manuscript's `p_ligand` as `p_gate` and
`p_binding` as `p_aiki` in the return dicts — same numbers, names
chosen for programmatic consistency with the function names.

**API default**: `score(peptide, allele)` returns the pure
`p_binding` factor (the allele-specific score). The manuscript's
recommended released-model output is the combined score, exposed
explicitly via `score_composite()`. Use `score_composite(...)` unless
your task is itself allele-specific.

## Quick start

```python
# The released model: combined presentation score (RECOMMENDED for most uses)
from aiki_hla import score_composite
out = score_composite(peptide="GILGFVFTL", allele="HLA-A*02:01", mode="both")
print(out["p_aiki"], out["p_gate"], out["p_composite"])
# 0.97  0.61  0.77  — combined = sqrt(0.97 * 0.61) = sqrt(p_binding * p_ligand)

# Pure allele-specific factor (cross-tool benchmarks, cluster-novel work)
from aiki_hla import score
p_binding = score(peptide="GILGFVFTL", allele="HLA-A*02:01")
print(p_binding)                               # 3-seed ensemble p_binding

# Gate only (epitope-likelihood; no MHC sequence, no torch)
from aiki_hla import score_gate
g = score_gate("GILGFVFTL", mhc_class="I")
print(g.p_gate, g.quality)                     # 0.61 ok
```

Command-line interface:

```bash
# Pure AIKI (default; the allele-specific factor only)
aiki-hla score --peptides peptides.csv --alleles alleles.csv --out predictions.csv

# Composite (replaces the score column with √(p_ligand · p_binding))
aiki-hla score --peptides peptides.csv --alleles alleles.csv \
               --mode presentation --out predictions.csv

# Both (emit p_aiki, p_gate, p_composite, gate_quality as separate columns)
aiki-hla score --peptides peptides.csv --alleles alleles.csv \
               --mode both --out predictions.csv

# Gate only (no AIKI, no torch needed)
aiki-hla gate --peptide GILGFVFTL --mhc-class I

aiki-hla evaluate --predictions P.csv --gold G.csv --strata broad,strict,both_novel
```

## Reproduce paper numbers

Every number in the manuscript is reproducible from the Zenodo deposit:

```bash
git clone https://github.com/aikium-public/aiki-hla.git
cd aiki-hla
python -m validation.reproduce_paper_numbers \
    --deposit https://doi.org/10.5281/zenodo.20520819
```

`reproduce_paper_numbers.py` loads each deposited result JSON, recomputes from
the same input artefacts, and asserts the published number within bootstrap
tolerance. The 3-seed ensemble checkpoints are SHA-256-pinned; a hash mismatch
aborts with "checkpoint identity mismatch; this is not v1.0".

## Repository layout

```
aiki_hla/
├── inference.py          load_trained_model, score, score_csv
├── train.py              training loop entrypoint
├── cli.py                argparse front-end for the aiki-hla command
├── data/                 corpus filters, splitting, embedding cache
├── models/               frozen-ESM head, pooling, attention
├── evaluation/           bootstrap CIs, tiered_metrics
├── training/             loss functions, samplers, callbacks
└── applications/         neoantigen, stability, off-target

scripts/                  user-facing CLIs (build corpus, fetch sources)
figures/                  figure-generation scripts (deterministic seeds)
configs/                  deployed-recipe YAML
validation/               leak_scan.py, reproduce_paper_numbers.py
checkpoints/README.md     points at Zenodo for the 3-seed ensemble
```

## Licenses

- Code: **Apache-2.0**
- Deposited corpus and result JSONs: **CC-BY-4.0**
- Fetch-only sources (NetMHCpan family, TransPHLA, VDJdb, MHCflurry training
  corpus, BigMHC training corpus): retain their upstream licenses; see
  `DATA_SOURCES.md` in the Zenodo deposit for fetch scripts and citations.

## Citation

If you use AIKI-HLA, please cite both the code and the manuscript:

```bibtex
@article{aikihla2026,
    title   = {Aiki-HLA: a factorized peptide-MHC predictor and a
               leakage-controlled benchmark for generalization to
               novel HLAs},
    author  = {Mysore, Venkatesh},
    journal = {Nature Machine Intelligence},
    year    = {2026},
    doi     = {10.5281/zenodo.20520820},
    note    = {Zenodo concept DOI 10.5281/zenodo.20520819 for the latest version},
}
```

## Reporting issues

GitHub Issues at https://github.com/aikium-public/aiki-hla/issues.
