"""URL model: character CNN -> BiLSTM -> attention pooling -> classifier.

    normalize -> char ids -> Embedding
              -> parallel Conv1d {3,5,7} -> concat
              -> BiLSTM
              -> additive attention pooling  (+ optional handcrafted branch)
              -> url_embedding (128) -> MLP head -> logit

Why this shape: the multi-kernel CNN captures local character n-grams (the
``.php``, ``login``, ``@`` patterns that dominate URL phishing), the BiLSTM
captures their order over a longer window, and attention pooling lets the model
weight the informative region of a long URL instead of averaging it away.

Outputs are returned as a dataclass so callers cannot accidentally confuse the
embedding with the logit. ``attention`` is retained because it is a secondary
explanation view alongside Integrated Gradients (P7).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass
class URLModelOutput:
    """Everything the URL model produces for one batch."""

    logit: Tensor  # (B,)
    probability: Tensor  # (B,) in [0, 1]
    embedding: Tensor  # (B, embedding_out)
    attention: Tensor  # (B, max_length) attention weights, sums to 1 per row


class AdditiveAttention(nn.Module):
    """Bahdanau-style additive attention pooling over the sequence.

    Masked positions receive ``-inf`` before the softmax so a padded tail
    contributes exactly nothing to the pooled representation.
    """

    def __init__(self, input_dim: int, attention_dim: int = 64) -> None:
        super().__init__()
        self.proj = nn.Linear(input_dim, attention_dim)
        self.score = nn.Linear(attention_dim, 1, bias=False)

    def forward(self, seq: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        """``seq``: (B, L, D), ``mask``: (B, L) with 1 = real, 0 = padding."""
        energy = self.score(torch.tanh(self.proj(seq))).squeeze(-1)  # (B, L)
        energy = energy.masked_fill(mask <= 0, float("-inf"))
        weights = torch.softmax(energy, dim=1)
        # A fully-padded row would produce NaNs; force uniform zeros-safe output.
        weights = torch.nan_to_num(weights, nan=0.0)
        pooled = torch.bmm(weights.unsqueeze(1), seq).squeeze(1)  # (B, D)
        return pooled, weights


class URLCharModel(nn.Module):
    """CharCNN + BiLSTM + attention URL classifier.

    Parameters follow ``configs/{dev,full}.yaml``. ``use_handcrafted`` toggles the
    separately-documented handcrafted branch; it is concatenated explicitly at the
    pooling stage so the two signals remain inspectable.
    """

    def __init__(
        self,
        vocab_size: int = 128,
        max_length: int = 256,
        embedding_dim: int = 64,
        cnn_channels: int = 48,
        cnn_kernel_sizes: tuple[int, ...] = (3, 5, 7),
        lstm_hidden: int = 48,
        lstm_layers: int = 1,
        bidirectional: bool = True,
        embedding_out: int = 128,
        dropout: float = 0.35,
        n_handcrafted: int = 0,
        use_handcrafted: bool = True,
    ) -> None:
        super().__init__()
        self.max_length = int(max_length)
        self.embedding_out = int(embedding_out)
        self.use_handcrafted = bool(use_handcrafted) and n_handcrafted > 0
        self.n_handcrafted = int(n_handcrafted) if self.use_handcrafted else 0

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        self.convolutions = nn.ModuleList(
            [
                nn.Conv1d(embedding_dim, cnn_channels, kernel_size=k, padding=k // 2)
                for k in cnn_kernel_sizes
            ]
        )
        conv_out = cnn_channels * len(cnn_kernel_sizes)

        self.lstm = nn.LSTM(
            input_size=conv_out,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )
        lstm_out = lstm_hidden * (2 if bidirectional else 1)

        self.attention = AdditiveAttention(lstm_out)

        # Length reduction from the conv stack, so pooling sees the LSTM's true
        # time steps.
        self.conv_reduction = sum(k - 1 for k in cnn_kernel_sizes) // len(cnn_kernel_sizes)

        pooled_dim = lstm_out
        if self.use_handcrafted:
            # Small dedicated trunk so the handcrafted signal is standardised
            # before it meets the pooled sequence representation.
            self.handcrafted = nn.Sequential(
                nn.Linear(self.n_handcrafted, 32),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            pooled_dim += 32

        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Sequential(
            nn.Linear(pooled_dim, embedding_out),
            nn.LayerNorm(embedding_out),
        )
        self.classifier = nn.Sequential(
            nn.Linear(embedding_out, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    # ------------------------------------------------------------------
    def forward(self, char_ids: Tensor, mask: Tensor, handcrafted: Tensor | None = None) -> URLModelOutput:
        """Run the model.

        Args:
            char_ids: ``(B, L)`` int64 character ids.
            mask: ``(B, L)`` float/int with 1 for real characters, 0 for padding.
            handcrafted: ``(B, n_handcrafted)`` standardised features. Required
                when ``use_handcrafted`` is True.

        Returns:
            :class:`URLModelOutput` with ``logit``, ``probability``,
            ``embedding`` (B, embedding_out) and ``attention`` (B, L).
        """
        if char_ids.dim() != 2:
            raise ValueError(f"char_ids must be (B, L), got {tuple(char_ids.shape)}")
        if mask.shape != char_ids.shape:
            raise ValueError(
                f"mask shape {tuple(mask.shape)} != char_ids shape {tuple(char_ids.shape)}"
            )

        emb = self.embedding(char_ids)  # (B, L, E)
        return self.forward_from_embedding(emb, mask, handcrafted)

    def forward_from_embedding(
        self,
        emb: Tensor,
        mask: Tensor,
        handcrafted: Tensor | None = None,
    ) -> URLModelOutput:
        """Run everything downstream of the character embedding.

        Split out from :meth:`forward` so explainability can run integrated
        gradients against the *embedding output* rather than the integer
        character ids. Attributing the ids is not possible: Captum interpolates
        between the input and a baseline, which makes the ids float, and
        ``torch.embedding`` only accepts Long/Int indices. Attributing in
        embedding space is the standard fix and gives a per-character score by
        summing the attribution across the embedding dimension.
        """
        if emb.dim() != 3:
            raise ValueError(f"emb must be (B, L, E), got {tuple(emb.shape)}")

        # Read the sequence length BEFORE transposing. After the swap
        # emb.size(1) is the embedding dimension, not the length, and using it
        # here silently trims the convolutions to 64 steps -- which only shows up
        # as a mask/energy shape mismatch deep inside the attention.
        seq_len = emb.size(1)

        emb = emb.transpose(1, 2)  # (B, E, L)

        # Parallel convolutions, each padded then trimmed back to L so the
        # attention weights stay aligned with the input characters.
        conv_outs: list[Tensor] = []
        for i, conv in enumerate(self.convolutions):
            c = conv(emb)
            c = c[:, :, :seq_len]
            if c.size(2) < seq_len:
                c = nn.functional.pad(c, (0, seq_len - c.size(2)))
            conv_outs.append(c)
        conv_cat = torch.cat(conv_outs, dim=1)  # (B, C*K, L)

        seq, _ = self.lstm(conv_cat.transpose(1, 2))  # (B, L, H)
        pooled, weights = self.attention(seq, mask)

        if self.use_handcrafted:
            if handcrafted is None:
                raise ValueError(
                    "this model was built with the handcrafted branch enabled, so "
                    "handcrafted features must be supplied"
                )
            if handcrafted.shape[1] != self.n_handcrafted:
                raise ValueError(
                    f"expected {self.n_handcrafted} handcrafted features, "
                    f"got {handcrafted.shape[1]}"
                )
            pooled = torch.cat([pooled, self.handcrafted(handcrafted)], dim=1)

        embedding = self.projection(self.dropout(pooled))  # (B, embedding_out)
        logit = self.classifier(self.dropout(embedding)).squeeze(-1)  # (B,)
        prob = torch.sigmoid(logit)  # (B,) in [0, 1]
        return URLModelOutput(
            logit=logit,
            probability=prob,
            embedding=embedding,
            attention=weights,
        )

    # ------------------------------------------------------------------
    def config(self) -> dict:
        """Architecture config saved into the checkpoint."""
        return {
            "vocab_size": self.embedding.num_embeddings,
            "max_length": self.max_length,
            "embedding_dim": self.embedding.embedding_dim,
            "cnn_channels": self.convolutions[0].out_channels,
            "cnn_kernel_sizes": [c.kernel_size[0] for c in self.convolutions],
            "lstm_hidden": self.lstm.hidden_size,
            "lstm_layers": self.lstm.num_layers,
            "bidirectional": self.lstm.bidirectional,
            "embedding_out": self.embedding_out,
            "n_handcrafted": self.n_handcrafted,
            "use_handcrafted": self.use_handcrafted,
        }