"""Mask-aware adaptive fusion network.

The three unimodal encoders (URL, HTML, vision) each produce an embedding.
Fusion is not a plain concatenation, because the modalities are frequently
*missing*: on this dataset most 2022 phishing hosts no longer resolve, so a
large fraction of rows arrive with only a URL.

Two mechanisms handle that, and both are load-bearing:

**Modality gating.** A small network reads the three embeddings *and* the three
availability flags, then emits a non-negative weight per modality via a
softmax. Weights are re-normalised over the *available* modalities only, so a
missing modality contributes nothing instead of dragging the fused vector
toward the zero embedding. The gates are returned with every prediction, which
is what the UI shows as "why this verdict".

**Masked modality dropout.** During training each available modality is dropped
independently with probability ``p``, so the fusion head cannot come to depend
on vision always being present. Without it, the gates collapse onto whatever
modality happens to be complete most often, and inference degrades sharply when
a different modality is the one that loaded.

Auxiliary unimodal heads are attached during training so each encoder keeps
producing a usable standalone score; the weight is small and configurable.
"""

from __future__ import annotations

import torch
from torch import nn

__all__ = ["FusionModel", "FusionOutput"]


class FusionOutput:
    __slots__ = ("logit", "gates", "modality_logits")

    def __init__(
        self,
        logit: torch.Tensor,
        gates: torch.Tensor,
        modality_logits: dict[str, torch.Tensor],
    ) -> None:
        self.logit = logit
        self.gates = gates
        self.modality_logits = modality_logits


class FusionModel(nn.Module):
    def __init__(
        self,
        shared_dim: int = 128,
        hidden_dims: tuple[int, ...] = (128, 64),
        dropout: float = 0.2,
        modality_dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.config = dict(
            shared_dim=shared_dim,
            hidden_dims=list(hidden_dims),
            dropout=dropout,
            modality_dropout=modality_dropout,
        )
        self.n_mod = 3

        # Project each modality into the shared space before anything mixes them.
        self.proj_url = nn.Sequential(
            nn.Linear(shared_dim, shared_dim), nn.LayerNorm(shared_dim), nn.GELU()
        )
        self.proj_html = nn.Sequential(
            nn.Linear(shared_dim, shared_dim), nn.LayerNorm(shared_dim), nn.GELU()
        )
        self.proj_vision = nn.Sequential(
            nn.Linear(shared_dim, shared_dim), nn.LayerNorm(shared_dim), nn.GELU()
        )

        # Gating sees embeddings + availability + pairwise agreement statistics.
        gate_in = shared_dim * self.n_mod + self.n_mod + 3
        self.gate = nn.Sequential(
            nn.Linear(gate_in, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, self.n_mod),
        )

        # Head input is [projected embeddings (3 * shared) | weighted fused vector
        # (shared) | availability flags (3)].
        self.mlp = nn.Sequential(
            nn.Linear(shared_dim * 4 + self.n_mod, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Linear(hidden_dims[1], 1)

        # Auxiliary heads so each encoder stays independently discriminative.
        self.aux_url = nn.Linear(shared_dim, 1)
        self.aux_html = nn.Linear(shared_dim, 1)
        self.aux_vision = nn.Linear(shared_dim, 1)

    # ------------------------------------------------------------------
    def _gates(self, embs: list[torch.Tensor], avail: torch.Tensor) -> torch.Tensor:
        """Availability-normalised modality weights."""
        pair_stats = torch.stack(
            [
                torch.cosine_similarity(embs[0], embs[1], dim=-1).abs(),
                torch.cosine_similarity(embs[0], embs[2], dim=-1).abs(),
                torch.cosine_similarity(embs[1], embs[2], dim=-1).abs(),
            ],
            dim=-1,
        )
        raw = torch.cat([torch.cat(embs, dim=-1), avail, pair_stats], dim=-1)
        logits = self.gate(raw)

        # Re-normalise over available modalities only. A missing modality must
        # get weight zero, not a share of whatever softmax would have given it.
        neg_inf = torch.finfo(logits.dtype).min
        masked = logits.masked_fill(avail <= 0, neg_inf)

        # A row with no modality at all would make softmax uniform-but-invalid;
        # fall back to the URL slot so the forward pass stays finite.
        none_available = (avail.sum(dim=-1, keepdim=True) <= 0)
        if none_available.any():
            fallback = torch.zeros_like(masked)
            fallback[:, 0] = 1.0
            masked = torch.where(none_available, fallback, masked)
        return torch.softmax(masked, dim=-1)

    def forward(
        self,
        emb_url: torch.Tensor,
        emb_html: torch.Tensor,
        emb_vision: torch.Tensor,
        mask_url: torch.Tensor,
        mask_html: torch.Tensor,
        mask_vision: torch.Tensor,
    ) -> FusionOutput:
        # Zero out unavailable modalities *before* projection, so a missing
        # branch contributes an exact zero rather than a projection of zeros
        # with an arbitrary bias.
        avail = torch.stack([mask_url, mask_html, mask_vision], dim=-1).to(emb_url.dtype)
        eu = self.proj_url(emb_url) * avail[:, 0:1]
        eh = self.proj_html(emb_html) * avail[:, 1:2]
        ev = self.proj_vision(emb_vision) * avail[:, 2:3]

        gates = self._gates([eu, eh, ev], avail)
        fused = gates[:, 0:1] * eu + gates[:, 1:2] * eh + gates[:, 2:3] * ev

        # The head also sees the raw availability flags, so it can condition on
        # *how much* evidence exists, not only on the weighted sum.
        mlp_in = torch.cat(
            [torch.cat([eu, eh, ev], dim=-1), fused, avail], dim=-1
        )
        logit = self.head(self.mlp(mlp_in)).squeeze(-1)

        aux = {
            "url": self.aux_url(eu).squeeze(-1),
            "html": self.aux_html(eh).squeeze(-1),
            "vision": self.aux_vision(ev).squeeze(-1),
        }
        return FusionOutput(logit=logit, gates=gates, modality_logits=aux)

    def config_dict(self) -> dict:
        return dict(self.config)
