"""ESM-2 650M per-residue embedding cache.

Computes ESM-2 embeddings on demand and caches them to disk so subsequent
calls on the same sequence return without re-running ESM. The cache is
keyed by the SHA-256 of the uppercased AA sequence.

Public API:
    EmbeddingCache(cache_dir, device='cpu', max_workers=1)
        .embed_peptide(peptide_str) -> torch.Tensor (L, 1280)
        .embed_mhc(g_domain_str)    -> torch.Tensor (L, 1280)
        .embed_batch(seqs)          -> list[torch.Tensor]

The first call materialises the ESM-2 650M model (~2.5 GB download via
`fair-esm`). Set `AIKI_MHC_CACHE` env var to override the on-disk location;
default is ``~/.cache/aiki-hla/embeddings/``.

The deployed AIKI-HLA head expects per-residue embeddings from
``esm2_t33_650M_UR50D`` representation layer 33 (i.e., the final
post-norm hidden state). Other layers are not used.
"""
from __future__ import annotations

import hashlib
import os
from functools import lru_cache
from pathlib import Path

import torch


ESM2_NAME = "esm2_t33_650M_UR50D"
ESM2_REPR_LAYER = 33
ESM2_DIM = 1280


def _sha256(seq: str) -> str:
    return hashlib.sha256(seq.upper().encode("utf-8")).hexdigest()


def _default_cache_dir() -> Path:
    env = os.environ.get("AIKI_MHC_CACHE")
    if env:
        return Path(env) / "embeddings"
    return Path.home() / ".cache" / "aiki-hla" / "embeddings"


@lru_cache(maxsize=1)
def _load_esm() -> tuple:
    """Lazy-load the ESM-2 650M model + alphabet (heavyweight; called once)."""
    import esm  # fair-esm
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model.eval()
    return model, alphabet


class EmbeddingCache:
    """On-disk + in-memory cache of ESM-2 per-residue embeddings.

    Each sequence's tensor is stored as a single .pt file under
    ``<cache_dir>/<sha256>.pt`` (FP16 for size). In-memory cache uses
    `lru_cache` (per-process; cleared on shutdown).
    """

    def __init__(self, cache_dir: str | Path | None = None, device: str = "cpu"):
        self.cache_dir = Path(cache_dir) if cache_dir else _default_cache_dir()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self._model = None
        self._alphabet = None
        self._batch_converter = None

    def _ensure_model(self) -> None:
        if self._model is None:
            model, alphabet = _load_esm()
            self._model = model.to(self.device)
            self._alphabet = alphabet
            self._batch_converter = alphabet.get_batch_converter()

    def _cache_path(self, seq: str) -> Path:
        return self.cache_dir / f"{_sha256(seq)}.pt"

    def _load_from_disk(self, seq: str) -> torch.Tensor | None:
        p = self._cache_path(seq)
        if p.exists():
            try:
                return torch.load(p, map_location=self.device, weights_only=True)
            except Exception:
                return None
        return None

    def _save_to_disk(self, seq: str, emb: torch.Tensor) -> None:
        torch.save(emb.to(torch.float16), self._cache_path(seq))

    def embed_batch(self, sequences: list[str]) -> list[torch.Tensor]:
        """Embed a batch of sequences. Uses disk cache where available.

        Returns one (L, 1280) tensor per input sequence (FP32 in memory).
        """
        out: list[torch.Tensor | None] = [None] * len(sequences)
        to_compute: list[tuple[int, str]] = []

        for i, seq in enumerate(sequences):
            cached = self._load_from_disk(seq)
            if cached is not None:
                out[i] = cached.to(torch.float32)
            else:
                to_compute.append((i, seq))

        if to_compute:
            self._ensure_model()
            batch_data = [(f"seq{i}", seq) for i, seq in to_compute]
            _, _, tokens = self._batch_converter(batch_data)
            tokens = tokens.to(self.device)
            with torch.no_grad():
                results = self._model(tokens, repr_layers=[ESM2_REPR_LAYER], return_contacts=False)
            reps = results["representations"][ESM2_REPR_LAYER]  # (B, L+2, 1280) — includes BOS/EOS
            for (i, seq), rep in zip(to_compute, reps):
                # Strip BOS (idx 0) and EOS (last); keep L residues
                L = len(seq)
                emb = rep[1:1 + L].cpu()
                self._save_to_disk(seq, emb)
                out[i] = emb.to(torch.float32)

        return out  # type: ignore[return-value]

    def embed_peptide(self, peptide: str) -> torch.Tensor:
        """Embed a single peptide string. Returns (L, 1280)."""
        return self.embed_batch([peptide])[0]

    def embed_mhc(self, g_domain_sequence: str) -> torch.Tensor:
        """Embed a single MHC G-domain sequence. Returns (L, 1280)."""
        return self.embed_batch([g_domain_sequence])[0]
