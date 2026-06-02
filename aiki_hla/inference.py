"""Inference API for the AIKI-HLA 3-seed deployment ensemble.

Public surface:
    score(peptide, allele, **kwargs)              -> float
        single-pair convenience function returning the ensemble probability.

    score_dataframe(df, peptide_col, allele_col)  -> pd.DataFrame
        scores every row of a DataFrame; adds a ``prob`` column.

    score_csv(peptides_path, alleles_path, out_path, **kwargs)
        CSV → CSV pipeline; reads two columns, writes per-row probabilities.

    load_trained_model(seeds='all', checkpoint_dir=None)
        loads the requested seeds' checkpoints into AikiHLA instances.
        Verifies each .pt against the pinned SHA-256 hashes before returning.

    ensure_embeddings(sequences, kind='peptide', cache=None)
        prefetches ESM-2 embeddings for a list of sequences (so subsequent
        scoring runs use the disk cache).

Defaults align with the manuscript:
    * 3-seed ensemble (seeds 42, 43, 44) averaged at the probability level
    * ESM-2 650M backbone (frozen; loaded on-demand)
    * batch_size 32 for the head forward pass; tunable
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from aiki_hla.data.sequences import get_g_domain_sequence, is_supported


# Canonical SHA-256 of each deployment-ensemble checkpoint, keyed by
# ensemble identifier (see Zenodo deposit's per-set MANIFEST.json files).
# Mismatch → abort.
#
# Two ensembles are released:
#   open-4.66M-421alleles      CC-BY-4.0  — pip-install default (commercial-OK).
#   research-use-5.8M-507alleles CC-BY-NC-4.0 — opt-in by passing
#       ``ensemble='research-use-5.8M-507alleles'``; served by Modal demo.
ENSEMBLE_CHECKPOINT_HASHES = {
    "open-4.66M-421alleles": {
        42: "08e72cfacca345b3b88dc929d3e2b497003b8627143670688d9b95468fe5a124",
        43: "5fda63043c895b5b5ae29a56a0cb4777bb5736cd7d6b5b4c605076cd1d4407be",
        44: "2a9822aeba70dd29875d503e76803cca99f05d7651607549972539c5e47128b1",
    },
    "research-use-5.8M-507alleles": {
        42: "f59c6bab58ea4d2898dbb2c86116fd18bd4886637442ec62134ec4a6cc6e6074",
        43: "0090a1f42d8324e7d248fb85ba0b3d0f55978c6e482977e1dc1268539812c91b",
        44: "60c316b3b6624cdbe4ff37e8c03c02399cf75a8b9d89e82b1fa9d6c087498fb0",
    },
}
DEFAULT_ENSEMBLE = "open-4.66M-421alleles"
# Back-compat alias retained for any caller that imports the old name.
CHECKPOINT_HASHES = ENSEMBLE_CHECKPOINT_HASHES[DEFAULT_ENSEMBLE]


def _default_cache_dir() -> Path:
    env = os.environ.get("AIKI_MHC_CACHE")
    if env:
        return Path(env) / "checkpoints"
    return Path.home() / ".cache" / "aiki-hla" / "checkpoints"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            buf = f.read(1 << 20)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def _verify_checkpoint(seed: int, path: Path, ensemble: str = DEFAULT_ENSEMBLE) -> None:
    """Raise if the file's SHA-256 doesn't match the pinned hash for `ensemble`."""
    actual = _sha256_file(path)
    expected = ENSEMBLE_CHECKPOINT_HASHES[ensemble][seed]
    if actual != expected:
        raise RuntimeError(
            f"Checkpoint identity mismatch for seed {seed} (ensemble={ensemble!r}).\n"
            f"  expected SHA-256: {expected}\n"
            f"  actual SHA-256:   {actual}\n"
            f"  path: {path}\n"
            f"This is NOT the released v1.0 {ensemble} checkpoint. Refusing to proceed."
        )


def load_trained_model(
    seeds: str | int | list[int] = "all",
    checkpoint_dir: str | Path | None = None,
    device: str = "cpu",
    ensemble: str = DEFAULT_ENSEMBLE,
) -> dict:
    """Load one or more seed checkpoints into AikiHLA instances.

    Parameters
    ----------
    seeds
        ``'all'`` (the 3-seed ensemble; default), an int (one seed), or a
        list of ints subset of {42, 43, 44}.
    checkpoint_dir
        Directory containing ``aiki_hla_seed_{42,43,44}.pt``. Defaults to
        ``$AIKI_MHC_CACHE/checkpoints/`` then to ``~/.cache/aiki-hla/checkpoints/``.
    device
        torch device ('cpu' or 'cuda:0' etc.).

    Returns
    -------
    dict
        Mapping ``{seed: AikiHLA}``. Each model is in eval mode and on the
        requested device. Use ``model(pep_emb, mhc_emb, pep_mask, mhc_mask)``
        to forward.

    Raises
    ------
    RuntimeError
        If any checkpoint's SHA-256 doesn't match the pinned hash.
    """
    from aiki_hla.models import AikiHLA, DEFAULT_RECIPE_KWARGS

    if ensemble not in ENSEMBLE_CHECKPOINT_HASHES:
        raise ValueError(
            f"Unknown ensemble: {ensemble!r}. "
            f"Supported: {sorted(ENSEMBLE_CHECKPOINT_HASHES)}"
        )
    ensemble_hashes = ENSEMBLE_CHECKPOINT_HASHES[ensemble]

    if seeds == "all":
        seeds_list = [42, 43, 44]
    elif isinstance(seeds, int):
        seeds_list = [seeds]
    else:
        seeds_list = list(seeds)
    for s in seeds_list:
        if s not in ensemble_hashes:
            raise ValueError(f"Unknown seed: {s}. Supported: {sorted(ensemble_hashes)}")

    cache_dir = Path(checkpoint_dir) if checkpoint_dir else _default_cache_dir()

    models: dict = {}
    for s in seeds_list:
        path = cache_dir / f"aiki_hla_seed_{s}.pt"
        if not path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {path}\n"
                f"Download the 3-seed deployment ensemble from the Zenodo concept DOI "
                f"and place under {cache_dir}, or pass --checkpoint-dir."
            )
        _verify_checkpoint(s, path, ensemble=ensemble)
        m = AikiHLA(**DEFAULT_RECIPE_KWARGS)
        sd = torch.load(path, map_location=device, weights_only=False)["model_state_dict"]
        m.load_state_dict(sd, strict=True)
        m.eval().to(device)
        models[s] = m
    return models


def _prepare_inputs(
    peptide: str,
    allele: str,
    cache,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Look up MHC sequence, embed both, build masks. Returns (pep_emb, mhc_emb, pep_mask, mhc_mask) batched B=1."""
    mhc_seq = get_g_domain_sequence(allele)
    if mhc_seq is None:
        raise ValueError(
            f"Allele {allele!r} is not in the bundled MHC sequence database. "
            f"Use aiki_hla.data.sequences.list_alleles() to see supported alleles, "
            f"or call aiki_hla.data.sequences.extend_database(...) with a downloaded "
            f"copy of the full Zenodo-hosted DB."
        )
    pep_emb = cache.embed_peptide(peptide).unsqueeze(0)
    mhc_emb = cache.embed_mhc(mhc_seq).unsqueeze(0)
    pep_mask = torch.ones((1, pep_emb.size(1)), dtype=torch.bool)
    mhc_mask = torch.ones((1, mhc_emb.size(1)), dtype=torch.bool)
    return pep_emb, mhc_emb, pep_mask, mhc_mask


def ensure_embeddings(sequences: list[str], kind: str = "peptide", cache=None) -> None:
    """Prefetch ESM-2 embeddings for a list of sequences (warms the disk cache)."""
    if cache is None:
        from aiki_hla.data.embedding_cache import EmbeddingCache
        cache = EmbeddingCache()
    cache.embed_batch(sequences)


def score(
    peptide: str,
    allele: str,
    seeds: str | list[int] = "all",
    checkpoint_dir: str | Path | None = None,
    device: str = "cpu",
    cache=None,
) -> float:
    """Score a single (peptide, allele) pair against the 3-seed ensemble.

    Returns
    -------
    float
        Calibrated binding probability in [0, 1]. The ensemble probability
        is the arithmetic mean of `sigmoid(logit)` across the requested seeds
        (default: all three).
    """
    if cache is None:
        from aiki_hla.data.embedding_cache import EmbeddingCache
        cache = EmbeddingCache(device=device)
    models = load_trained_model(seeds=seeds, checkpoint_dir=checkpoint_dir, device=device)
    pep_emb, mhc_emb, pep_mask, mhc_mask = _prepare_inputs(peptide, allele, cache)
    probs = []
    with torch.no_grad():
        for _, m in models.items():
            out = m(pep_emb.to(device), mhc_emb.to(device), pep_mask.to(device), mhc_mask.to(device))
            probs.append(torch.sigmoid(out["logits"]).cpu().item())
    return float(np.mean(probs))


def score_dataframe(
    df,
    peptide_col: str = "peptide",
    allele_col: str = "allele",
    seeds: str | list[int] = "all",
    checkpoint_dir: str | Path | None = None,
    device: str = "cpu",
    batch_size: int = 32,
):
    """Score every row of `df`. Returns the same df with a new `prob` column.

    Embeddings are pre-fetched in bulk (one ESM-2 pass for all unique peptides +
    unique MHC sequences) then each row is forwarded through the head ensemble
    in batches of `batch_size`.
    """
    import pandas as pd  # imported here to keep the package import-light

    from aiki_hla.data.embedding_cache import EmbeddingCache

    rows = df.copy()
    if peptide_col not in rows.columns or allele_col not in rows.columns:
        raise KeyError(f"DataFrame must have columns {peptide_col!r} and {allele_col!r}")

    unsupported = rows.loc[~rows[allele_col].map(is_supported), allele_col].unique()
    if len(unsupported) > 0:
        raise ValueError(
            f"Unsupported alleles: {sorted(unsupported)[:10]}"
            f"{'…' if len(unsupported) > 10 else ''}"
        )

    cache = EmbeddingCache(device=device)
    unique_peptides = rows[peptide_col].astype(str).unique().tolist()
    unique_mhc_seqs = [get_g_domain_sequence(a) for a in rows[allele_col].astype(str).unique().tolist()]
    cache.embed_batch(unique_peptides)
    cache.embed_batch(unique_mhc_seqs)

    models = load_trained_model(seeds=seeds, checkpoint_dir=checkpoint_dir, device=device)

    probs = np.empty(len(rows))
    for start in range(0, len(rows), batch_size):
        end = min(start + batch_size, len(rows))
        sub = rows.iloc[start:end]
        pep_embs = cache.embed_batch(sub[peptide_col].astype(str).tolist())
        mhc_embs = cache.embed_batch([get_g_domain_sequence(a) for a in sub[allele_col].astype(str).tolist()])
        # Pad to common length within the batch
        L_pep = max(e.size(0) for e in pep_embs)
        L_mhc = max(e.size(0) for e in mhc_embs)
        B = len(pep_embs)
        pep_batch = torch.zeros(B, L_pep, 1280)
        mhc_batch = torch.zeros(B, L_mhc, 1280)
        pep_mask = torch.zeros(B, L_pep, dtype=torch.bool)
        mhc_mask = torch.zeros(B, L_mhc, dtype=torch.bool)
        for i, (pe, me) in enumerate(zip(pep_embs, mhc_embs)):
            pep_batch[i, :pe.size(0)] = pe
            mhc_batch[i, :me.size(0)] = me
            pep_mask[i, :pe.size(0)] = True
            mhc_mask[i, :me.size(0)] = True
        seed_probs = []
        with torch.no_grad():
            for _, m in models.items():
                out = m(pep_batch.to(device), mhc_batch.to(device), pep_mask.to(device), mhc_mask.to(device))
                seed_probs.append(torch.sigmoid(out["logits"]).cpu().numpy())
        probs[start:end] = np.mean(seed_probs, axis=0)

    rows["prob"] = probs
    return rows


def score_csv(
    peptides_path: str,
    alleles_path: str,
    out_path: str,
    peptide_col: str = "peptide",
    allele_col: str = "allele",
    seeds: str | list[int] = "all",
    checkpoint_dir: str | Path | None = None,
    device: str = "cpu",
) -> None:
    """CSV → CSV: score every (peptide, allele) row and write out a `prob` column."""
    import pandas as pd

    peps = pd.read_csv(peptides_path)
    alls = pd.read_csv(alleles_path)
    if len(peps) != len(alls):
        raise ValueError(f"row count mismatch: peptides={len(peps)} alleles={len(alls)}")
    df = peps.copy()
    df[allele_col] = alls[allele_col].values
    scored = score_dataframe(
        df, peptide_col=peptide_col, allele_col=allele_col,
        seeds=seeds, checkpoint_dir=checkpoint_dir, device=device,
    )
    scored.to_csv(out_path, index=False)
