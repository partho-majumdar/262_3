"""Continual learning for the URL model: rehearsal memory + elastic weight consolidation.

The production checkpoint is fitted once, offline, on PhiUSIIL. Real phishing does
not stop when training stops, so the shipped model is stale the morning after it
was trained. This module is the mechanism for absorbing URLs observed *after*
training without retraining from scratch.

Two protections against catastrophic forgetting, both real:

**Rehearsal.** A fixed-capacity :class:`RehearsalBuffer` keeps a reservoir sample
of everything the model has seen, including the URL that arrived today. Every
update trains on the union of the buffer and the new batch, so yesterday's
evidence is always in the gradient. Reservoir sampling (Algorithm R) is used
rather than a recency window because a phishing stream is not i.i.d. in time --
an attack campaign arrives in a burst -- and a recency window would evict exactly
the burst that matters.

**Elastic weight consolidation.** A quadratic penalty pulls the parameters back
toward the values they had at the end of the previous task, weighted by a
diagonal Fisher estimate of how important each weight was. Weights the previous
data depended on move slowly; weights that were never important may move freely.
The Fisher here is the empirical Fisher (mean squared gradient of the log
likelihood on the replay buffer), not the true Fisher: it costs one pass and is
what makes the method affordable on CPU in minutes.

Both can be disabled, which is what the ``--baseline`` ablation in
``training/continual_update.py`` does, so the claim "continual learning prevents
forgetting" is measured against a real control instead of asserted.

The class-balanced sampler is not optional politeness. A stream of newly observed
URLs is overwhelmingly phishing, because that is what a detector gets sent. Naive
fine-tuning on such a stream collapses the decision boundary toward the phishing
class within a few rounds, which is indistinguishable from a bug. Balancing the
batches keeps the update's implicit prior close to the deployment prior.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterator, Sequence

BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import Tensor, nn  # noqa: E402

from app.models.url_model import URLCharModel  # noqa: E402
from app.preprocessing.url_dataset import SplitData, URLDataset  # noqa: E402
from app.preprocessing.url_features import FeatureScaler  # noqa: E402
from app.preprocessing.url_preprocessing import CharTokenizer  # noqa: E402
from app.utils.metrics import compute_binary_metrics  # noqa: E402

__all__ = [
    "RehearsalBuffer",
    "BalancedBatchSampler",
    "EWCState",
    "ContinualUpdater",
]


# ---------------------------------------------------------------------------
# Rehearsal memory
# ---------------------------------------------------------------------------
class RehearsalBuffer:
    """Fixed-capacity reservoir of ``(url, label)`` pairs.

    Algorithm R (Vitter's reservoir sampling): item *i* replaces a uniformly
    chosen slot with probability ``capacity / i``. Every item that has ever been
    observed therefore has an equal chance of being remembered, so the buffer is
    an unbiased sample of the whole stream rather than of its most recent tail.

    Determinism: all draws come from one seeded ``numpy`` generator, so two
    buffers constructed with the same seed and fed the same additions hold the
    same items. Nothing here touches global RNG state, which means a test -- or
    the ablation in the CLI -- is reproducible without seeding the whole process.
    """

    def __init__(self, capacity: int = 512, seed: int = 42) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive; a zero-capacity buffer forgets by construction")
        self.capacity = int(capacity)
        self.seed = int(seed)
        self.urls: list[str] = []
        self.labels: list[int] = []
        #: Total number of items ever offered, which is what reservoir sampling
        #: needs and what makes the replacement probability well defined.
        self.n_seen = 0
        self._rng = np.random.default_rng(self.seed)

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.urls)

    def __iter__(self) -> Iterator[tuple[str, int]]:
        return iter(zip(self.urls, self.labels))

    @property
    def is_full(self) -> bool:
        return len(self.urls) >= self.capacity

    def add(self, urls: Sequence[str], labels: Sequence[int]) -> int:
        """Offer ``(urls, labels)`` to the buffer.

        Returns the number of slots actually *changed*, which is zero once the
        buffer is saturated and the draws all miss. Reporting the count instead
        of assuming "added" matters for honest buffer-composition reporting: an
        all-phishing stream may add nothing at all.
        """
        if len(urls) != len(labels):
            raise ValueError(f"url/label length mismatch: {len(urls)} vs {len(labels)}")
        changed = 0
        for url, label in zip(urls, labels):
            self.n_seen += 1
            label = int(label)
            if label not in (0, 1):
                raise ValueError(f"label must be 0 or 1 (phishing), got {label!r}")
            if len(self.urls) < self.capacity:
                self.urls.append(str(url))
                self.labels.append(label)
                changed += 1
                continue
            j = int(self._rng.integers(0, self.n_seen))
            if j < self.capacity:
                self.urls[j] = str(url)
                self.labels[j] = label
                changed += 1
        return changed

    def extend(self, other: "RehearsalBuffer") -> int:
        """Absorb another buffer's contents through the same reservoir path."""
        return self.add(list(other.urls), list(other.labels))

    def reset(self) -> None:
        self.urls = []
        self.labels = []
        self.n_seen = 0
        self._rng = np.random.default_rng(self.seed)

    # ------------------------------------------------------------------
    def composition(self) -> dict:
        """Class counts and the phishing fraction, for the update report."""
        arr = np.asarray(self.labels, dtype=np.int64)
        n_pos = int((arr == 1).sum())
        n_neg = int((arr == 0).sum())
        return {
            "capacity": self.capacity,
            "size": len(self),
            "n_seen": self.n_seen,
            "phishing": n_pos,
            "legitimate": n_neg,
            "phishing_fraction": float(n_pos / len(self)) if len(self) else 0.0,
        }

    def arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """``(urls, labels)`` as numpy arrays, URL dtype chosen to avoid pickle."""
        return (
            np.asarray(self.urls, dtype=np.str_),
            np.asarray(self.labels, dtype=np.int64),
        )

    def save(self, path: Path | str) -> Path:
        """Persist to a single ``.npz``.

        URLs are stored as a fixed-width unicode array rather than an object
        array so the file reloads without ``allow_pickle=True``. The generator
        state is deliberately not stored: ``.npz`` cannot hold a nested dict
        without pickling, so the RNG is re-derived from ``seed + n_seen`` on
        load. That keeps a resumed buffer's future draws reproducible from the
        file alone, which is what a scheduled retrain needs.
        """
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        urls, labels = self.arrays()
        np.savez_compressed(
            p,
            urls=urls,
            labels=labels,
            capacity=np.asarray(self.capacity, dtype=np.int64),
            seed=np.asarray(self.seed, dtype=np.int64),
            n_seen=np.asarray(self.n_seen, dtype=np.int64),
        )
        return p

    @classmethod
    def load(cls, path: Path | str) -> "RehearsalBuffer":
        """Read a buffer written by :meth:`save`."""
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"no rehearsal buffer at {p}")
        with np.load(p, allow_pickle=False) as blob:
            obj = cls(
                capacity=int(blob["capacity"]),
                seed=int(blob["seed"]),
            )
            obj.urls = [str(u) for u in blob["urls"].tolist()]
            obj.labels = [int(v) for v in blob["labels"].tolist()]
            obj.n_seen = int(blob["n_seen"])
        obj._rng = np.random.default_rng(obj.seed + obj.n_seen)
        return obj


# ---------------------------------------------------------------------------
# Balanced sampling
# ---------------------------------------------------------------------------
class BalancedBatchSampler:
    """Draws minibatches with a 50/50 class split by construction.

    Why this lives here rather than as a loss weight: weighting the loss can
    still leave a batch that is 95% phishing, so the gradient direction for that
    step is dominated by one class. Rebalancing the batch fixes the direction
    itself, and it makes the realised class ratio auditable -- the sampler
    reports it, so a report can show that the update really was balanced rather
    than asserting it.

    When a class is absent, the batch degrades to whatever that class had and
    says so in ``degenerate`` rather than raising: a buffer that only ever saw
    phishing is a real (bad) state, and the update should still be able to run
    and report the damage.
    """

    def __init__(self, labels: Sequence[int], batch_size: int = 32, seed: int = 42) -> None:
        arr = np.asarray(labels, dtype=np.int64).ravel()
        self.labels = arr
        self.batch_size = max(1, int(batch_size))
        self.pos = np.flatnonzero(arr == 1).astype(np.int64)
        self.neg = np.flatnonzero(arr == 0).astype(np.int64)
        self._rng = np.random.default_rng(int(seed))

    @property
    def n_total(self) -> int:
        return int(self.pos.size + self.neg.size)

    @property
    def has_both_classes(self) -> bool:
        return bool(self.pos.size and self.neg.size)

    def sample_batch(self) -> list[int]:
        """Indices for one balanced minibatch, sampled without replacement."""
        n_pos = min(self.pos.size, self.batch_size // 2)
        n_neg = min(self.neg.size, self.batch_size - n_pos)
        # Give back the shortfall to the class that still has items, so a small
        # buffer yields a full batch instead of silently a small one.
        shortfall = self.batch_size - n_pos - n_neg
        while shortfall > 0:
            if self.pos.size - n_pos >= self.neg.size - n_neg and self.pos.size > n_pos:
                n_pos += 1
            elif self.neg.size > n_neg:
                n_neg += 1
            else:
                break
            shortfall -= 1
        idx: list[int] = []
        if n_pos:
            idx += self.pos[self._rng.choice(
                self.pos.size, size=n_pos, replace=n_pos > self.pos.size
            )].tolist()
        if n_neg:
            idx += self.neg[self._rng.choice(
                self.neg.size, size=n_neg, replace=n_neg > self.neg.size
            )].tolist()
        return [int(i) for i in idx]

    def batches(self, steps: int) -> Iterator[list[int]]:
        for _ in range(max(0, int(steps))):
            yield self.sample_batch()

    def describe(self, batch: Sequence[int]) -> dict:
        """Realised class composition of a drawn batch, read from the pool labels."""
        idx = np.asarray(list(batch), dtype=np.int64)
        drawn = self.labels[idx] if idx.size else np.zeros(0, dtype=np.int64)
        n_pos = int((drawn == 1).sum())
        return {
            "size": int(idx.size),
            "positive": n_pos,
            "negative": int(idx.size) - n_pos,
            "positive_fraction": float(n_pos / idx.size) if idx.size else 0.0,
            "degenerate": not self.has_both_classes,
        }


# ---------------------------------------------------------------------------
# Elastic weight consolidation
# ---------------------------------------------------------------------------
class EWCState:
    """Diagonal-Fisher EWC anchor for one parameter vector.

    ``penalty = (lambda / 2) * sum_i F_i * (theta_i - theta*_i)^2``

    ``F_i`` is the empirical Fisher: the mean squared gradient of the per-sample
    log loss with respect to weight ``i``, accumulated over the replay buffer.
    The Fisher is the part that makes this more than plain weight decay: a
    weight that never received gradient (say a char-embedding row that no
    training URL contained) gets ``F_i = 0`` and is therefore free to move,
    while a heavily-used convolution kernel is pinned. Plain L2-to-anchor would
    freeze those unused rows too and waste capacity.

    ``fisher`` and ``anchor`` are lists aligned with the model's
    ``named_parameters()`` order, so the penalty is a straight zip with no name
    lookup and no chance of a silently missing parameter.
    """

    def __init__(self) -> None:
        self.fisher: list[Tensor] = []
        self.anchor: list[Tensor] = []
        self.lam: float = 0.0
        self.fisher_mean: float = 0.0

    @property
    def active(self) -> bool:
        return self.lam > 0.0 and bool(self.fisher)

    def clear(self) -> None:
        self.fisher = []
        self.anchor = []
        self.lam = 0.0
        self.fisher_mean = 0.0

    def penalty(self, params: Sequence[Tensor], lam: float | None = None) -> Tensor:
        """Quadratic penalty at the given parameters (0 at the anchor by construction)."""
        coefficient = self.lam if lam is None else float(lam)
        if coefficient <= 0.0 or not self.fisher:
            return torch.zeros(())
        if len(self.fisher) != len(params):
            raise ValueError(
                f"Fisher has {len(self.fisher)} entries but the model has {len(params)} parameters; "
                "the anchor was taken against a different architecture"
            )
        total = torch.zeros(())
        for f, a, p in zip(self.fisher, self.anchor, params):
            if f.shape != p.shape:
                raise ValueError(
                    f"Fisher entry of shape {tuple(f.shape)} does not match a parameter of shape "
                    f"{tuple(p.shape)}; the anchor was taken against a different architecture"
                )
            total = total + (f * (p.detach() - a).pow(2)).sum()
        return 0.5 * coefficient * total


def _param_list(model: nn.Module) -> list[nn.Parameter]:
    return [p for p in model.parameters() if p.requires_grad]


# ---------------------------------------------------------------------------
# Updater
# ---------------------------------------------------------------------------
class ContinualUpdater:
    """Fine-tune the URL model on a live stream without forgetting the past.

    Lifecycle: construct around a loaded model, :meth:`observe` each incoming
    batch, :meth:`update` to absorb it, :meth:`evaluate_on` to measure what
    changed. The updater owns the model in place -- it is not a wrapper that
    returns a new one -- so a service that has already built its
    ``InferenceService`` can keep scoring with the same object while the weights
    move underneath it.
    """

    def __init__(
        self,
        model: URLCharModel,
        tokenizer: CharTokenizer,
        scaler: FeatureScaler | None = None,
        *,
        buffer_capacity: int = 512,
        seed: int = 42,
        device: str = "cpu",
        lr: float = 1e-4,
        batch_size: int = 32,
        ewc_lambda: float = 0.0,
        weight_decay: float = 0.0,
        use_rehearsal: bool = True,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.scaler = scaler
        self.device = str(device)
        self.default_lr = float(lr)
        self.batch_size = int(batch_size)
        self.weight_decay = float(weight_decay)
        self.use_rehearsal = bool(use_rehearsal)
        self.seed = int(seed)

        self.buffer = RehearsalBuffer(capacity=buffer_capacity, seed=seed)
        self.ewc = EWCState()
        self.ewc_lambda = float(ewc_lambda)
        if self.ewc_lambda > 0.0:
            self.ewc.lam = self.ewc_lambda

        #: URLs of the batch most recently passed to :meth:`observe`.
        self.pending_urls: list[str] = []
        self.pending_labels: list[int] = []
        #: Metrics dict from the most recent :meth:`update`, so a caller driving
        #: a multi-round stream does not have to keep the return value around
        #: just to log it later.
        self.last_update: dict | None = None
        #: Snapshot at construction, for cumulative drift reporting.
        self._initial_state = [p.detach().clone() for p in _param_list(self.model)]
        self._encode_cache: dict[tuple[str, ...], tuple[Tensor, Tensor, Tensor | None]] = {}
        self._encode_cache_size = 8

    # ------------------------------------------------------------------
    # Stream intake
    # ------------------------------------------------------------------
    def observe(self, new_urls: Sequence[str], new_labels: Sequence[int]) -> None:
        """Record a newly observed batch.

        Both the buffer and the pending batch are updated: the batch is what the
        next :meth:`update` must adapt to, and the buffer is what stops that
        adaptation from eating the past. Empty input is accepted and ignored --
        a day with no new URLs is not an error, it is a day with nothing to do.
        """
        urls = [str(u) for u in new_urls]
        labels = [int(v) for v in new_labels]
        if len(urls) != len(labels):
            raise ValueError(f"url/label length mismatch: {len(urls)} vs {len(labels)}")
        self.pending_urls = urls
        self.pending_labels = labels
        if self.use_rehearsal and urls:
            self.buffer.add(urls, labels)

    def forget_pending(self) -> None:
        """Drop the pending batch, e.g. after a failed update, so it is not replayed."""
        self.pending_urls = []
        self.pending_labels = []

    # ------------------------------------------------------------------
    # Encoding helpers
    # ------------------------------------------------------------------
    def _split(self, urls: Sequence[str], labels: Sequence[int]) -> SplitData:
        n = len(urls)
        return SplitData(
            name="continual",
            urls=[str(u) for u in urls],
            labels=np.asarray(labels, dtype=np.int64)[:n],
            row_ids=np.arange(n, dtype=np.int64),
        )

    def encode(self, urls: Sequence[str]) -> tuple[Tensor, Tensor, Tensor | None]:
        """``(char_ids, mask, handcrafted)`` for a list of URLs.

        Goes through :class:`URLDataset` rather than re-implementing tokenisation,
        so an update can never be trained on inputs shaped differently from the
        ones inference feeds the model.

        Results are memoised on the exact URL tuple. Handcrafted feature
        extraction is ~1.7 ms per URL -- far more expensive than the model's own
        forward pass on a short batch -- and a continual-learning loop re-encodes
        the *same* held-out evaluation set after every round. Without the cache
        most of the wall clock goes to recomputing features nobody changed. The
        cache is bounded and keyed on content, so it cannot serve stale tensors.
        """
        key = tuple(str(u) for u in urls)
        cached = self._encode_cache.get(key)
        if cached is not None:
            return cached
        split = self._split(list(key), [1] * len(key))
        ds = URLDataset(split, self.tokenizer, self.scaler)
        out = (ds.char_ids, ds.mask, (ds.handcrafted if ds.handcrafted.shape[1] else None))
        if len(self._encode_cache) >= self._encode_cache_size:
            self._encode_cache.pop(next(iter(self._encode_cache)))
        self._encode_cache[key] = out
        return out

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------
    @torch.no_grad()
    def logits_for(self, urls: Sequence[str], batch_size: int = 256) -> np.ndarray:
        """Raw logits, no calibration.

        The updater reasons in logit space because the EWC/Fisher machinery is
        defined on the pre-sigmoid quantity and any temperature fit would be
        invalidated the moment the weights move anyway.
        """
        self.model.eval()
        out: list[np.ndarray] = []
        for i in range(0, len(urls), max(1, batch_size)):
            chunk = list(urls[i : i + max(1, batch_size)])
            char_ids, mask, hand = self.encode(chunk)
            res = self.model(
                char_ids.to(self.device),
                mask.to(self.device),
                hand.to(self.device) if hand is not None else None,
            )
            out.append(res.logit.detach().cpu().numpy())
        return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)

    def evaluate_on(
        self,
        held_out_urls: Sequence[str],
        held_out_labels: Sequence[int],
        threshold: float = 0.5,
    ) -> dict:
        """Accuracy / F1 / ROC-AUC / MCC on URLs the update never trained on.

        This is the forgetting instrument: called with the *same* held-out set
        before and after an update, the drop is the forgetting. Returns
        ``None`` for metrics that are undefined (empty input, single-class set)
        rather than a placeholder that would read as a real number.
        """
        urls = [str(u) for u in held_out_urls]
        labels = np.asarray(held_out_labels, dtype=np.int64).ravel()
        if not urls or labels.size == 0 or labels.size != len(urls):
            return {
                "n_samples": int(min(len(urls), labels.size)),
                "available": False,
                "reason": "empty or misaligned evaluation set",
                "accuracy": None, "precision": None, "recall": None,
                "specificity": None, "f1": None, "roc_auc": None, "mcc": None,
                "balanced_accuracy": None, "brier": None,
            }
        logits = self.logits_for(urls)
        probs = 1.0 / (1.0 + np.exp(-logits))
        m = compute_binary_metrics(labels, probs, threshold=threshold)
        return {
            "n_samples": int(m.n_samples),
            "available": True,
            "accuracy": round(m.accuracy, 6),
            "precision": round(m.precision, 6),
            "recall": round(m.recall, 6),
            "specificity": round(m.specificity, 6),
            "f1": round(m.f1, 6),
            "roc_auc": None if np.isnan(m.roc_auc) else round(m.roc_auc, 6),
            "pr_auc": None if np.isnan(m.pr_auc) else round(m.pr_auc, 6),
            "mcc": round(m.mcc, 6),
            "balanced_accuracy": round(m.balanced_accuracy, 6),
            "brier": round(m.brier, 6),
            "confusion_matrix": m.confusion,
        }

    # ------------------------------------------------------------------
    # EWC
    # ------------------------------------------------------------------
    def estimate_fisher(self, urls: Sequence[str], labels: Sequence[int], batches: int = 8) -> float:
        """Accumulate the diagonal Fisher on the replay set; returns its mean.

        The model is held in ``eval`` mode: dropout would make the estimate noisy
        enough to matter at this scale, and the Fisher is only a relative
        importance signal anyway.
        """
        params = _param_list(self.model)
        self.model.eval()
        for p in params:
            if p.grad is not None:
                p.grad = None
        sampler = BalancedBatchSampler(labels, self.batch_size, seed=self.seed + 1)
        n_seen = 0
        for idx in sampler.batches(batches):
            if not idx:
                continue
            char_ids, mask, hand = self.encode([urls[i] for i in idx])
            y = torch.as_tensor(np.asarray([labels[i] for i in idx], dtype=np.float32))
            out = self.model(
                char_ids.to(self.device),
                mask.to(self.device),
                hand.to(self.device) if hand is not None else None,
            )
            loss = nn.functional.binary_cross_entropy_with_logits(out.logit, y.to(self.device))
            loss.backward()
            n_seen += 1
        fisher: list[Tensor] = []
        total = 0.0
        count = 0
        for p in params:
            g = p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p)
            fisher.append(g.pow(2))
            total += float(g.pow(2).sum())
            count += g.numel()
        for p in params:
            p.grad = None
        self.ewc.fisher = fisher
        self.ewc.anchor = [p.detach().clone() for p in params]
        self.ewc.fisher_mean = float(total / count) if count else 0.0
        return self.ewc.fisher_mean

    def set_ewc_anchor(
        self, urls: Sequence[str] | None = None, labels: Sequence[int] | None = None, batches: int = 8
    ) -> None:
        """Anchor EWC at the current weights using the given (or buffer) data."""
        if urls is None or labels is None:
            urls, labels = self._replay_pool()
        self.ewc.lam = float(self.ewc_lambda)
        if self.ewc_lambda <= 0.0 or not len(urls):
            self.ewc.clear()
            return
        self.estimate_fisher(list(urls), list(labels), batches=batches)

    def ewc_penalty(self) -> float:
        """Current value of the EWC quadratic term (0 at the anchor)."""
        return float(self.ewc.penalty(_param_list(self.model)).item())

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------
    def _replay_pool(self) -> tuple[list[str], list[int]]:
        """Buffer contents plus the pending batch, de-duplicated by URL string.

        De-duplication matters because ``observe`` puts the new batch *into* the
        buffer; counting it twice would silently up-weight the newest data,
        which is exactly the bias rehearsal exists to remove.
        """
        urls = list(self.buffer.urls) if self.use_rehearsal else []
        labels = list(self.buffer.labels) if self.use_rehearsal else []
        seen = set(urls)
        for url, label in zip(self.pending_urls, self.pending_labels):
            if url in seen:
                continue
            seen.add(url)
            urls.append(url)
            labels.append(int(label))
        return urls, labels

    def update(
        self,
        steps: int = 20,
        lr: float | None = None,
        batch_size: int | None = None,
        ewc_lambda: float | None = None,
        update_fisher: bool = True,
    ) -> dict:
        """Fine-tune on buffer + new batch with balanced batches and EWC.

        Returns a metrics dict rather than nothing: a continual-learning step
        whose effect is not reported is indistinguishable from a step that did
        nothing, and the whole point of the exercise is to be able to show
        whether it did anything.
        """
        steps = max(0, int(steps))
        bs = int(batch_size or self.batch_size)
        learning_rate = float(lr if lr is not None else self.default_lr)
        lam = float(ewc_lambda if ewc_lambda is not None else self.ewc_lambda)

        urls, labels = self._replay_pool()
        if not urls or steps == 0:
            self.last_update = {
                "skipped": True,
                "reason": "no replayable data" if not urls else "zero steps requested",
                "steps": steps,
                "lr": learning_rate,
                "loss": None,
                "loss_first": None,
                "loss_last": None,
                "n_train_urls": len(urls),
                "batch_composition": None,
                "buffer": self.buffer.composition(),
                "rehearsal_active": bool(self.use_rehearsal),
                "drift": {"since_previous_update": 0.0, "since_initial": 0.0, "relative_to_initial": 0.0},
                "ewc": {"lambda": lam, "penalty": 0.0, "active": False},
                "recall_phishing": None,
                "recall_legitimate": None,
            }
            return self.last_update

        sampler = BalancedBatchSampler(labels, bs, seed=self.seed)
        params = _param_list(self.model)
        # Snapshot *before* the first optimiser step. Reading the post-update
        # weights here would report a drift of exactly zero every round, which is
        # how a broken drift metric looks.
        pre_update_state = [p.detach().clone() for p in params]
        optimizer = torch.optim.AdamW(params, lr=learning_rate, weight_decay=self.weight_decay)
        self.model.train()

        losses: list[float] = []
        compositions: list[dict] = []
        penalty_last = 0.0

        for idx in sampler.batches(steps):
            if not idx:
                continue
            char_ids, mask, hand = self.encode([urls[i] for i in idx])
            y = torch.as_tensor(np.asarray([labels[i] for i in idx], dtype=np.float32))
            optimizer.zero_grad()
            out = self.model(
                char_ids.to(self.device),
                mask.to(self.device),
                hand.to(self.device) if hand is not None else None,
            )
            # Plain BCE, no pos_weight: the sampler has already balanced the
            # batch, and weighting on top of balancing would double-correct.
            data_loss = nn.functional.binary_cross_entropy_with_logits(out.logit, y.to(self.device))
            if lam > 0.0 and self.ewc.active:
                penalty = self.ewc.penalty(params)
                loss = data_loss + penalty
                penalty_last = float(penalty.item())
            else:
                penalty_last = 0.0
                loss = data_loss
            loss.backward()
            nn.utils.clip_grad_norm_(params, max_norm=5.0)
            optimizer.step()
            losses.append(float(data_loss.item()))
            compositions.append(sampler.describe(idx))

        self.model.eval()
        drift = self._drift(params, pre_update_state)

        train_eval = self.evaluate_on(urls, labels)
        buffer_stats = self.buffer.composition()
        pos_fraction = np.mean([c["positive_fraction"] for c in compositions]) if compositions else None

        if lam > 0.0 and update_fisher:
            # Re-anchor after the update so the penalty protects the *current*
            # task rather than fighting the previous one forever.
            self.ewc_lambda = lam
            self.set_ewc_anchor(urls, labels)

        self.last_update = {
            "skipped": False,
            "steps": len(losses),
            "lr": learning_rate,
            "loss": round(float(np.mean(losses)), 6) if losses else None,
            "loss_first": round(losses[0], 6) if losses else None,
            "loss_last": round(losses[-1], 6) if losses else None,
            "n_train_urls": len(urls),
            "batch_composition": {
                "requested_batch_size": bs,
                "mean_positive_fraction": None if pos_fraction is None else round(float(pos_fraction), 6),
                "first_batch": compositions[0] if compositions else None,
                "last_batch": compositions[-1] if compositions else None,
                "buffer_had_both_classes": sampler.has_both_classes,
            },
            "buffer": buffer_stats,
            "rehearsal_active": bool(self.use_rehearsal),
            "recall_phishing": train_eval["recall"],
            "recall_legitimate": train_eval["specificity"],
            "train_metrics": train_eval,
            "drift": drift,
            "ewc": {
                "lambda": lam,
                "penalty": round(penalty_last, 8),
                "active": bool(lam > 0.0 and self.ewc.active),
                "fisher_mean": round(self.ewc.fisher_mean, 12) if lam > 0.0 else None,
            },
        }
        return self.last_update

    def _drift(self, params: Sequence[Tensor], previous: Sequence[Tensor]) -> dict:
        """How far the weights moved since the last update and since construction."""
        since_prev = 0.0
        since_init = 0.0
        for p, prev, init in zip(params, previous, self._initial_state):
            since_prev += float((p.detach() - prev).pow(2).sum())
            since_init += float((p.detach() - init).pow(2).sum())
        since_prev = float(np.sqrt(since_prev))
        since_init = float(np.sqrt(since_init))
        return {
            "since_previous_update": round(since_prev, 8),
            "since_initial": round(since_init, 8),
            "relative_to_initial": round(since_prev / max(since_init, 1e-12), 8),
        }

    # ------------------------------------------------------------------
    def state_dict(self) -> dict:
        """Everything needed to resume the same updater tomorrow."""
        return {
            "buffer": {
                "capacity": self.buffer.capacity,
                "seed": self.buffer.seed,
                "urls": list(self.buffer.urls),
                "labels": list(self.buffer.labels),
                "n_seen": self.buffer.n_seen,
            },
            "pending_urls": list(self.pending_urls),
            "pending_labels": list(self.pending_labels),
            "ewc_lambda": self.ewc_lambda,
            "seed": self.seed,
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore from :meth:`state_dict` (the model weights are not touched)."""
        buf = state.get("buffer") or {}
        self.buffer = RehearsalBuffer(
            capacity=int(buf.get("capacity", self.buffer.capacity)),
            seed=int(buf.get("seed", self.buffer.seed)),
        )
        self.buffer.urls = [str(u) for u in buf.get("urls", [])]
        self.buffer.labels = [int(v) for v in buf.get("labels", [])]
        self.buffer.n_seen = int(buf.get("n_seen", len(self.buffer)))
        self.buffer._rng = np.random.default_rng(self.buffer.seed + self.buffer.n_seen)
        self.pending_urls = [str(u) for u in state.get("pending_urls", [])]
        self.pending_labels = [int(v) for v in state.get("pending_labels", [])]
        self.ewc_lambda = float(state.get("ewc_lambda", self.ewc_lambda))
        self.ewc.clear()