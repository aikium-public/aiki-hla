"""AIKI-HLA model architecture.

The deployed predictor is a frozen ESM-2 650M backbone with a
3.9M-parameter (3,941,834 trainable parameters) head consisting of:
    * 1280 → 256 projection (peptide + MHC, separate weights)
    * 1 self-attention layer per side (8 heads, ff_mult=4)
    * position-biased multi-head attention pooling on the peptide side
    * pooling-by-multi-head-attention (PMA) on the MHC side
    * 512 → 256 → 256 fusion + 3-layer classifier MLP
    * auxiliary regression head on continuous IC50 measurements

The single trainable module is `AikiHLA`. It consumes pre-computed ESM-2
embeddings (the backbone is never updated during training); the loader
in `aiki_hla.inference` ensures the cache is warm before scoring.

Public surface:
    AikiHLA                       — the head module
    DEFAULT_RECIPE_KWARGS         — constructor kwargs matching the
                                    deployed 3-seed ensemble
"""
from aiki_hla.models.head import AikiHLA, DEFAULT_RECIPE_KWARGS  # noqa: F401

__all__ = ["AikiHLA", "DEFAULT_RECIPE_KWARGS"]
