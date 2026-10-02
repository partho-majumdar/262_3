"""Word-level tokenizer for the visible text of a fetched page.

Deliberately simple: lowercased whitespace/punctuation tokens with a frequency
cutoff. The alternative - a learned subword vocabulary - would need far more
page data than the collection budget on this host allows, and would mostly buy
a larger embedding table rather than better signal.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Sequence

import numpy as np

__all__ = ["HTMLTextTokenizer"]

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9'\-\.]*")

PAD, UNK = "<pad>", "<unk>"


class HTMLTextTokenizer:
    """Fixed-vocabulary word tokenizer with a stored state dict."""

    def __init__(self, vocab_size: int = 4096, max_tokens: int = 512, min_count: int = 2) -> None:
        self.vocab_size = int(vocab_size)
        self.max_tokens = int(max_tokens)
        self.min_count = int(min_count)
        self.itos: list[str] = [PAD, UNK]
        self.stoi: dict[str, int] = {PAD: 0, UNK: 1}

    @property
    def pad_id(self) -> int:
        return 0

    @property
    def unk_id(self) -> int:
        return 1

    @property
    def size(self) -> int:
        return len(self.itos)

    @staticmethod
    def tokenize(text: str) -> list[str]:
        return _TOKEN_RE.findall(text.lower())

    def fit(self, texts: Sequence[str]) -> "HTMLTextTokenizer":
        counts: Counter[str] = Counter()
        for t in texts:
            counts.update(self.tokenize(t))
        for tok, n in counts.most_common():
            if len(self.itos) >= self.vocab_size:
                break
            if n >= self.min_count and tok not in self.stoi:
                self.stoi[tok] = len(self.itos)
                self.itos.append(tok)
        return self

    def encode(self, text: str) -> tuple[list[int], list[int]]:
        ids = [self.stoi.get(t, self.unk_id) for t in self.tokenize(text)][: self.max_tokens]
        n = len(ids)
        pad = self.max_tokens - n
        return ids + [self.pad_id] * pad, [1] * n + [0] * pad

    def encode_batch(self, texts: Sequence[str]) -> tuple[list[list[int]], list[list[int]]]:
        pairs = [self.encode(t) for t in texts]
        return [p[0] for p in pairs], [p[1] for p in pairs]

    def state_dict(self) -> dict:
        return {
            "vocab_size": self.vocab_size,
            "max_tokens": self.max_tokens,
            "min_count": self.min_count,
            "itos": self.itos,
        }

    @classmethod
    def from_state_dict(cls, state: dict) -> "HTMLTextTokenizer":
        obj = cls(
            vocab_size=int(state["vocab_size"]),
            max_tokens=int(state["max_tokens"]),
            min_count=int(state.get("min_count", 2)),
        )
        obj.itos = list(state["itos"])
        obj.stoi = {tok: i for i, tok in enumerate(obj.itos)}
        return obj

    def save(self, path: Path) -> None:
        import json

        Path(path).write_text(json.dumps(self.state_dict()), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "HTMLTextTokenizer":
        import json

        return cls.from_state_dict(json.loads(Path(path).read_text(encoding="utf-8")))
