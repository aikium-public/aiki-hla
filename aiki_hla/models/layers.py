"""Reusable building blocks for AIKI-HLA.

This module ships only the layers that appear in the deployed 3-seed
ensemble's state_dict. Cross-attention blocks (PreNormCrossAttention) are
not part of the deployed recipe and are not included.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class PreNormSelfAttention(nn.Module):
    """Pre-LayerNorm self-attention block with residual connections.

    Matches `mhc_encoder.0` and `pep_encoder.0` in the deployed checkpoints:

        norm1 (LayerNorm)  →  attn (MultiheadAttention 8h)  →  residual
        norm2 (LayerNorm)  →  ff (Linear → GELU → Dropout → Linear → Dropout)  →  residual
    """

    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.1, ff_mult: int = 4):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True,
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ff_mult, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + attn_out
        x = x + self.ff(self.norm2(x))
        return x
