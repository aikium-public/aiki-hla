"""The deployed AIKI-HLA head module.

This implements exactly the forward pass that produced the 3-seed
deployment ensemble shipped with the v1.0 release. The state_dict keys of
the deployed checkpoints map 1:1 to the modules defined here; loading
should produce zero `missing_keys` and zero `unexpected_keys`.

Recipe (from the deployed `config` field in each checkpoint):
    esm_dim          = 1280
    d_model          = 256
    n_self_layers    = 1
    n_cross_layers   = 0
    n_heads          = 8
    ff_mult          = 4
    dropout          = 0.2
    pep_pool         = position-biased MHA (pep_pool_type 'anchor_aware')
    mhc_pool         = PMA (pooling-by-multi-head-attention, 1 seed)
    use_regression_head = True (auxiliary IC50 regression branch)

The auxiliary regression head shares the fused 256-D representation; at
inference it is computed but not used to produce the binding probability
(which comes from the classifier).
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from aiki_hla.models.layers import PreNormSelfAttention
from aiki_hla.models.pooling import PMA, PositionBiasedMHAPooling


DEFAULT_RECIPE_KWARGS: dict = {
    "esm_dim": 1280,
    "d_model": 256,
    "n_self_layers": 1,
    "n_heads": 8,
    "ff_mult": 4,
    "dropout": 0.2,
    "max_pep_len": 64,
    "max_mhc_len": 768,
    "use_regression_head": True,
}


class AikiHLA(nn.Module):
    """The 3.9M-parameter peptide × MHC binding head (3,941,834 trainable parameters; bit-identical state_dict to the released ensemble).

    Inputs to `forward`:
        pep_emb:   (B, L_pep, esm_dim)   peptide per-residue ESM-2 embeddings
        mhc_emb:   (B, L_mhc, esm_dim)   MHC G-domain per-residue embeddings
        pep_mask:  (B, L_pep)            True for valid positions
        mhc_mask:  (B, L_mhc)            True for valid positions

    Returns a dict with:
        'logits':   (B,)  binary classification logit (binding probability after sigmoid)
        'ba_pred':  (B,)  auxiliary regression output (continuous IC50 predictor)
                          — only present when use_regression_head=True
        'pep_repr': (B, d_model)
        'mhc_repr': (B, d_model)
    """

    def __init__(
        self,
        esm_dim: int = 1280,
        d_model: int = 256,
        n_self_layers: int = 1,
        n_heads: int = 8,
        ff_mult: int = 4,
        dropout: float = 0.2,
        max_pep_len: int = 64,
        max_mhc_len: int = 768,
        use_regression_head: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.use_regression_head = use_regression_head

        # --- Input projections (1280 → 256) with LayerNorm + Dropout ---
        # Sequential index map for state_dict keys:
        #   .0 = Linear(esm_dim, d_model)
        #   .1 = GELU            (no params)
        #   .2 = Dropout         (no params)
        #   .3 = Linear(d_model, d_model)
        #   .4 = LayerNorm(d_model)
        self.pep_proj = nn.Sequential(
            nn.Linear(esm_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )
        self.mhc_proj = nn.Sequential(
            nn.Linear(esm_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )

        # --- Positional + chain-type embeddings ---
        self.pep_pos = nn.Embedding(max_pep_len, d_model)
        self.mhc_pos = nn.Embedding(max_mhc_len, d_model)
        self.chain_type_embed = nn.Embedding(2, d_model)
        self._pos_scale = d_model ** -0.5

        # --- Per-side self-attention encoders (one layer each in deployed) ---
        self.pep_encoder = nn.ModuleList([
            PreNormSelfAttention(d_model, n_heads, dropout, ff_mult)
            for _ in range(n_self_layers)
        ])
        self.mhc_encoder = nn.ModuleList([
            PreNormSelfAttention(d_model, n_heads, dropout, ff_mult)
            for _ in range(n_self_layers)
        ])

        # --- Post-stack norms ---
        # The post_sa norm applies right after self-attention; the post_ca norm
        # applies after cross-attention. Since the deployed recipe has
        # n_cross_layers=0 there is no cross-attention loop, but the post_ca
        # norm is still applied (unconditional pre-pool stabilisation).
        self.pep_post_sa_norm = nn.LayerNorm(d_model)
        self.mhc_post_sa_norm = nn.LayerNorm(d_model)
        self.pep_post_ca_norm = nn.LayerNorm(d_model)
        self.mhc_post_ca_norm = nn.LayerNorm(d_model)

        # --- Pooling ---
        self.pep_pool = PositionBiasedMHAPooling(d_model, max_len=max_pep_len, n_heads=n_heads, dropout=dropout)
        self.mhc_pool = PMA(d_model, n_heads=n_heads, n_seeds=1, dropout=dropout)

        # --- Fusion: cat(pep_pooled, mhc_pooled) → 256 → 256 ---
        # Sequential index map:
        #   .0 = Linear(2 * d_model, d_model)
        #   .1 = GELU
        #   .2 = Dropout
        #   .3 = Linear(d_model, d_model)
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )

        # --- Classification head (3-layer MLP: 256 → 128 → 64 → 1) ---
        # Sequential index map:
        #   .0 = Linear(d_model,     d_model // 2)
        #   .1 = GELU
        #   .2 = Dropout
        #   .3 = Linear(d_model // 2, d_model // 4)
        #   .4 = GELU
        #   .5 = Dropout
        #   .6 = Linear(d_model // 4, 1)
        self.final_norm = nn.LayerNorm(d_model)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, 1),
        )

        # --- Auxiliary regression head (continuous IC50) ---
        if use_regression_head:
            self.regressor = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, 1),
            )

    def forward(
        self,
        pep_emb: torch.Tensor,
        mhc_emb: torch.Tensor,
        pep_mask: torch.Tensor,
        mhc_mask: torch.Tensor,
    ) -> dict:
        B = pep_emb.size(0)

        # --- Project to d_model ---
        pep = self.pep_proj(pep_emb)  # (B, L_pep, d_model)
        mhc = self.mhc_proj(mhc_emb)  # (B, L_mhc, d_model)

        # --- Positional encoding (scaled) ---
        pep_positions = torch.arange(pep.size(1), device=pep.device).unsqueeze(0).expand(B, -1)
        mhc_positions = torch.arange(mhc.size(1), device=mhc.device).unsqueeze(0).expand(B, -1)
        pep = pep + self.pep_pos(pep_positions) * self._pos_scale
        mhc = mhc + self.mhc_pos(mhc_positions) * self._pos_scale

        # --- Chain-type embedding (peptide=0, MHC=1) ---
        pep = pep + self.chain_type_embed(torch.zeros(1, dtype=torch.long, device=pep.device)) * self._pos_scale
        mhc = mhc + self.chain_type_embed(torch.ones(1, dtype=torch.long, device=mhc.device)) * self._pos_scale

        # --- Self-attention ---
        pep_pad = ~pep_mask
        mhc_pad = ~mhc_mask
        for layer in self.pep_encoder:
            pep = layer(pep, key_padding_mask=pep_pad)
        for layer in self.mhc_encoder:
            mhc = layer(mhc, key_padding_mask=mhc_pad)

        # --- Post-self-attention norm ---
        pep = self.pep_post_sa_norm(pep)
        mhc = self.mhc_post_sa_norm(mhc)

        # --- Post-cross-attention norm (unconditional; cross-attn is absent in deployed) ---
        pep = self.pep_post_ca_norm(pep)
        mhc = self.mhc_post_ca_norm(mhc)

        # --- Pool to fixed-size representations ---
        pep_pooled = self.pep_pool(pep, pep_mask)
        mhc_pooled = self.mhc_pool(mhc, mhc_mask)

        # --- Fuse + classify ---
        fused = self.fusion(torch.cat([pep_pooled, mhc_pooled], dim=-1))
        fused = self.final_norm(fused)
        logits = self.classifier(fused).squeeze(-1)

        out: dict = {
            "logits": logits,
            "pep_repr": pep_pooled,
            "mhc_repr": mhc_pooled,
        }
        if self.use_regression_head:
            out["ba_pred"] = self.regressor(fused).squeeze(-1)
        return out
