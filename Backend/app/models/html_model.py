"""Page-content (HTML) modality model.

Two branches over one document:

* a small Transformer encoder over the **visible text** tokens, which is where
  credential-harvesting language ("verify your identity", "session expired")
  shows up; and
* an MLP over the **structured DOM features** from
  :mod:`app.preprocessing.html_features` (form counts, password fields, brand
  vs host mismatch, external form actions, and so on).

The branches are concatenated into a single embedding that the fusion network
consumes. When a document is missing the caller passes the availability mask and
the fused model knows to disregard this branch.
"""

from __future__ import annotations

import torch
from torch import nn

__all__ = ["HTMLModalityModel", "HTMLModelOutput"]


class HTMLModelOutput:
    """Container so callers can use the same attribute style as the URL model."""

    __slots__ = ("embedding", "logit")

    def __init__(self, embedding: torch.Tensor, logit: torch.Tensor) -> None:
        self.embedding = embedding
        self.logit = logit


class HTMLModalityModel(nn.Module):
    def __init__(
        self,
        vocab_size: int = 4096,
        max_tokens: int = 512,
        n_dom_features: int = 45,
        embedding_dim: int = 128,
        transformer_layers: int = 2,
        transformer_heads: int = 4,
        transformer_ff: int = 256,
        embedding_out: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.config = dict(
            vocab_size=vocab_size,
            max_tokens=max_tokens,
            n_dom_features=n_dom_features,
            embedding_dim=embedding_dim,
            transformer_layers=transformer_layers,
            transformer_heads=transformer_heads,
            transformer_ff=transformer_ff,
            embedding_out=embedding_out,
            dropout=dropout,
        )

        self.token_emb = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.pos_emb = nn.Embedding(max_tokens, embedding_dim)
        self.text_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=transformer_heads,
                dim_feedforward=transformer_ff,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
            ),
            num_layers=transformer_layers,
        )
        self.text_proj = nn.Linear(embedding_dim, embedding_out)

        self.dom_mlp = nn.Sequential(
            nn.Linear(n_dom_features, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, embedding_out),
        )

        self.fuse = nn.Sequential(
            nn.Linear(embedding_out * 2, embedding_out),
            nn.LayerNorm(embedding_out),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Linear(embedding_out, 1)

    @staticmethod
    def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Mean over real tokens only.

        Without this, padding positions dominate the average and the model reads
        document *length* rather than content.
        """
        m = mask.unsqueeze(-1).to(x.dtype)
        denom = m.sum(dim=1).clamp(min=1.0)
        return (x * m).sum(dim=1) / denom

    def forward(
        self,
        token_ids: torch.Tensor,
        token_mask: torch.Tensor,
        dom_feats: torch.Tensor,
    ) -> HTMLModelOutput:
        if token_ids.numel() == 0:
            # No text branch available; fall back to the DOM branch alone.
            emb = self.dom_mlp(dom_feats)
            return HTMLModelOutput(embedding=emb, logit=self.head(emb).squeeze(-1))

        positions = torch.arange(token_ids.size(1), device=token_ids.device)
        h = self.token_emb(token_ids) + self.pos_emb(positions).unsqueeze(0)
        pad_mask = token_mask == 0
        # A fully-masked row would make softmax produce NaN; let those rows
        # attend freely and rely on the masked mean to discard them.
        pad_mask = pad_mask & ~pad_mask.all(dim=1, keepdim=True)
        h = self.text_encoder(h, src_key_padding_mask=pad_mask)
        text = self.text_proj(self._masked_mean(h, token_mask))
        emb = self.fuse(torch.cat([text, self.dom_mlp(dom_feats)], dim=-1))
        return HTMLModelOutput(embedding=emb, logit=self.head(emb).squeeze(-1))

    def config_dict(self) -> dict:
        return dict(self.config)
