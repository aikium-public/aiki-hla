"""Command-line front-end for `aiki-hla`.

    aiki-hla score    --peptides X.csv --alleles Y.csv [--mode binding|presentation|both]
                      [--seed 42|43|44|all] --out P.csv
    aiki-hla evaluate --predictions P.csv --gold G.csv --strata broad,strict,both_novel
    aiki-hla lookup   --peptide AAAAA --allele "HLA-A*02:01" [--mode binding|presentation|both]
    aiki-hla gate     --peptide AAAAA --mhc-class I|II

The CLI is a thin wrapper around the module API. It fetches the 3-seed
ensemble checkpoints from the Zenodo concept-DOI on first call and caches
them under ~/.cache/aiki-hla/. The viability-gate pickles ship as
package data and need no network fetch.

``--mode`` regime selection (see ``aiki_hla.viability_gate``):

  - ``binding``       — pure AIKI per-(peptide, allele) probability
                        (default). Use for allele-specific cluster-novel
                        discrimination.
  - ``presentation``  — composite ``sqrt(p_aiki * p_gate)``. Use for
                        off-target screening, neoantigen ranking, and
                        proteome-wide scoring.
  - ``both``          — emit p_aiki, p_gate, p_composite, and a quality
                        flag in the output CSV; let the caller pick.
"""
from __future__ import annotations

import argparse
import sys


def _score_cmd(args) -> int:
    """Batch CSV scoring with optional gate composition."""
    import pandas as pd
    from aiki_hla.inference import score_csv

    # Always run the AIKI pass; CSV pred path is unchanged at mode=binding.
    score_csv(
        peptides_path=args.peptides,
        alleles_path=args.alleles,
        out_path=args.out,
        seeds=args.seed,
    )
    if args.mode == "binding":
        return 0
    # Augment the output CSV with gate + composite columns.
    from aiki_hla.viability_gate import score_gate_batch, compose
    df = pd.read_csv(args.out)
    if "peptide" not in df.columns or "allele" not in df.columns:
        sys.stderr.write(
            "ERROR: --mode requires the output CSV to carry peptide + allele "
            "columns (check that score_csv wrote them).\n"
        )
        return 2
    gates = score_gate_batch(
        peptides=df["peptide"].astype(str).str.upper().tolist(),
        alleles=df["allele"].astype(str).tolist(),
    )
    df["p_gate"] = [g.p_gate for g in gates]
    df["gate_quality"] = [g.quality for g in gates]
    # The AIKI probability column is conventionally called "binding_prob"
    # by inference.score_csv. Compose against it.
    p_aiki_col = next(
        (c for c in ("binding_prob", "score", "prob") if c in df.columns),
        None,
    )
    if p_aiki_col is None:
        sys.stderr.write(
            "ERROR: no AIKI probability column found in output CSV "
            "(expected one of: binding_prob, score, prob).\n"
        )
        return 2
    df["p_composite"] = [
        compose(float(pa), float(pg))
        for pa, pg in zip(df[p_aiki_col].values, df["p_gate"].values)
    ]
    if args.mode == "presentation":
        # Replace the headline AIKI column with the composite for callers
        # who only want one score; keep p_aiki / p_gate / p_composite for
        # transparency.
        df = df.rename(columns={p_aiki_col: "p_aiki"})
    df.to_csv(args.out, index=False)
    return 0


def _evaluate_cmd(args) -> int:
    from aiki_hla.evaluation.tiered_metrics import compute_tiered_metrics
    import pandas as pd
    preds = pd.read_csv(args.predictions)
    gold = pd.read_csv(args.gold)
    strata = args.strata.split(",")
    results = compute_tiered_metrics(preds, gold, strata=strata)
    import json
    print(json.dumps(results, indent=2))
    return 0


def _lookup_cmd(args) -> int:
    if args.mode == "binding":
        from aiki_hla.inference import score
        prob = score(peptide=args.peptide, allele=args.allele)
        print(f"{args.peptide}  {args.allele}  {prob:.6f}")
        return 0
    from aiki_hla.viability_gate import score_composite
    out = score_composite(peptide=args.peptide, allele=args.allele, mode=args.mode)
    pa = "    None" if out["p_aiki"] is None else f"{out['p_aiki']:.6f}"
    pc = "    None" if out["p_composite"] is None else f"{out['p_composite']:.6f}"
    print(
        f"{args.peptide}  {args.allele}  "
        f"p_aiki={pa}  p_gate={out['p_gate']:.6f}  p_composite={pc}  "
        f"class={out['mhc_class']}  quality={out['quality']}"
    )
    return 0


def _gate_cmd(args) -> int:
    """Gate-only score (no AIKI; no MHC sequence needed)."""
    from aiki_hla.viability_gate import score_gate
    g = score_gate(args.peptide, mhc_class=args.mhc_class, allele=args.allele)
    print(
        f"{args.peptide}  class={g.mhc_class}  p_gate={g.p_gate:.6f}  "
        f"quality={g.quality}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="aiki-hla", description="AIKI-HLA pMHC binding predictor")
    ap.add_argument("--version", action="version", version="aiki-hla 1.0.0")
    sub = ap.add_subparsers(dest="command", required=True)

    sc = sub.add_parser("score", help="Score peptide x allele pairs from CSVs")
    sc.add_argument("--peptides", required=True, help="CSV with a peptide column")
    sc.add_argument("--alleles", required=True, help="CSV with an allele column")
    sc.add_argument("--out", required=True, help="output CSV path")
    sc.add_argument("--seed", default="all", help="42, 43, 44, or 'all' for the 3-seed ensemble (default)")
    sc.add_argument(
        "--mode", choices=["binding", "presentation", "both"], default="binding",
        help="Regime: 'binding' (pure AIKI; default); "
             "'presentation' (replace AIKI col with composite "
             "sqrt(p_ligand * p_binding); use for proteome screening); "
             "'both' (emit p_aiki / p_gate / p_composite / gate_quality "
             "as separate columns).",
    )
    sc.set_defaults(func=_score_cmd)

    ev = sub.add_parser("evaluate", help="Evaluate predictions against gold")
    ev.add_argument("--predictions", required=True)
    ev.add_argument("--gold", required=True)
    ev.add_argument("--strata", default="broad,strict,both_novel")
    ev.set_defaults(func=_evaluate_cmd)

    lk = sub.add_parser("lookup", help="Single peptide x allele lookup")
    lk.add_argument("--peptide", required=True)
    lk.add_argument("--allele", required=True)
    lk.add_argument(
        "--mode", choices=["binding", "presentation", "both"], default="binding",
        help="See `aiki-hla score --mode` for the regime contract.",
    )
    lk.set_defaults(func=_lookup_cmd)

    gt = sub.add_parser(
        "gate",
        help="Peptide-only viability gate (no MHC; no torch). "
             "Cheap presentation-likeness pre-filter.",
    )
    gt.add_argument("--peptide", required=True)
    gt.add_argument(
        "--mhc-class", choices=["I", "II"], default=None,
        help="Force a class; otherwise inferred from --allele or peptide length.",
    )
    gt.add_argument(
        "--allele", default=None,
        help="Optional; used only to infer the class when --mhc-class is omitted.",
    )
    gt.set_defaults(func=_gate_cmd)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
