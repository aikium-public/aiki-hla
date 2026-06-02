"""Pooling modules used in the AIKI-HLA deployed architecture.

Two poolers:
    * PositionBiasedMHAPooling — multi-head attention pooling with a
      learnable per-absolute-position scalar bias. Applied to the
      peptide side (`pep_pool` in the checkpoint).
    * PMA (Pooling by Multi-head Attention, Lee et al. 2019) — a single
      learnable seed vector attends over the input set. Applied to the
      MHC side (`mhc_pool`).

The `pep_pool` checkpoint sub-tree uses the keys `combine`, `content_proj`,
`norm`, `pos_bias`. The `mhc_pool` sub-tree uses `attn`, `ff`, `norm`,
`norm2`, `seeds`. The class definitions below match those keys exactly.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionBiasedMHAPooling(nn.Module):
    """Position-biased multi-head attention pooling.

    Each token gets a content-based attention weight (per head) plus a
    learnable scalar position bias indexed by absolute position. Weights
    are softmaxed over the sequence dimension within each head; per-head
    pooled vectors are concatenated and projected back to d_model.

    The pos_bias parameter is empirically near-decorative on the deployed
    recipe (Methods §"Model, training, and ensemble") but retained for
    checkpoint compatibility.
    """

    def __init__(
        self,
        d_model: int,
        max_len: int = 64,
        n_heads: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        # Learnable per-absolute-position scalar bias, zero-initialised
        self.pos_bias = nn.Parameter(torch.zeros(max_len))
        # Content-based attention scores: (d_model) → (d_model/2) → (n_heads)
        self.content_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, n_heads),
        )
        self.n_heads = n_heads
        # Multi-head pooling concatenates head outputs (n_heads × d_model) and
        # projects back to d_model. Only present when n_heads > 1.
        if n_heads > 1:
            self.combine = nn.Linear(d_model * n_heads, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, D) token embeddings
            mask: (B, L) True for valid tokens, False for padding
        Returns:
            (B, D) pooled representation
        """
        B, L, D = x.shape
        content_scores = self.content_proj(x)  # (B, L, n_heads)
        pos_bias = self.pos_bias[:L].unsqueeze(0).unsqueeze(-1)  # (1, L, 1)
        scores = content_scores + pos_bias  # broadcast
        scores = scores.masked_fill(~mask.unsqueeze(-1), -1e9)
        weights = F.softmax(scores, dim=1)  # (B, L, n_heads)

        if self.n_heads == 1:
            pooled = (x * weights).sum(dim=1)
        else:
            heads = []
            for h in range(self.n_heads):
                w = weights[:, :, h:h + 1]
                heads.append((x * w).sum(dim=1))
            pooled = self.combine(torch.cat(heads, dim=-1))
        return self.norm(pooled)


class PMA(nn.Module):
    """Pooling by Multi-head Attention (Set Transformer, Lee et al. 2019).

    A single learnable seed vector (shape (1, n_seeds, d_model)) serves as
    the query in a multi-head attention layer over the input set, producing
    a fixed-size output regardless of input length. Followed by a residual
    feedforward block.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        n_seeds: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.seeds = nn.Parameter(torch.randn(1, n_seeds, d_model) * 0.02)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True,
        )
        self.norm = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.n_seeds = n_seeds

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, D)
            mask: (B, L) True for valid
        Returns:
            (B, D) if n_seeds == 1, else (B, n_seeds * D)
        """
        B = x.size(0)
        seeds = self.seeds.expand(B, -1, -1)
        attn_out, _ = self.attn(seeds, x, x, key_padding_mask=~mask, need_weights=False)
        h = self.norm(seeds + attn_out)
        h = self.norm2(h + self.ff(h))
        if self.n_seeds == 1:
            return h.squeeze(1)
        return h.reshape(B, -1)
