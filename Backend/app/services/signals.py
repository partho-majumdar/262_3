"""Live signals that sit alongside the three modelled modalities.

The URL, HTML and screenshot branches each have a trained checkpoint. The
signals collected here -- TLS certificate metadata, live JavaScript behaviour,
page layout geometry and brand-image impersonation -- are computed from an
actual observation of the target rather than from a stored corpus, which means
there is no labelled crawl to train a classifier head on. They are therefore
reported as measured features with an explicit availability flag and reason,
not folded into the fusion as if a trained model had scored them.

Reporting them honestly is the point: a marker can see exactly what was
observed, and the block makes the gap between "we collect this" and "a model
learns from this" visible instead of implied.

The graph branch is different -- ``training/train_gnn.py`` produced a real
checkpoint -- so it returns a genuine model probability. Its measured strength
is modest (+0.045 accuracy over a majority-class baseline, MCC 0.286); see
``reports/metrics_gnn.md`` for why, and do not present it as competitive with
the URL branch.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import structlog

log = structlog.get_logger("signals")

BACKEND_ROOT = Path(__file__).resolve().parents[2]


class SignalBlock:
    """One signal group: availability, why it is missing, and its features."""

    __slots__ = ("available", "reason", "features", "probability")

    def __init__(
        self,
        available: bool,
        reason: str | None = None,
        features: dict[str, float] | None = None,
        probability: float | None = None,
    ) -> None:
        self.available = available
        self.reason = reason
        self.features = features or {}
        self.probability = probability

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"available": self.available}
        if self.reason is not None:
            out["reason"] = self.reason
        if self.probability is not None:
            out["probability"] = round(float(self.probability), 6)
        if self.features:
            out["features"] = {k: round(float(v), 6) for k, v in self.features.items()}
        return out


def _named(vector: list[float], names) -> dict[str, float]:
    return {n: float(v) for n, v in zip(names, vector)}


def collect_certificate(url: str, *, timeout: float = 5.0) -> SignalBlock:
    """TLS certificate features for the URL's host.

    Blocking (it opens a socket), so callers must run this off the event loop.
    """
    from app.preprocessing.cert_features import (
        CERT_FEATURE_NAMES,
        extract_cert_features_for_url,
        ssl_error_reason,
    )

    try:
        vec = extract_cert_features_for_url(url, timeout=timeout)
    except Exception as exc:  # noqa: BLE001 - a signal must never break /analyze
        return SignalBlock(False, reason=f"{exc.__class__.__name__}: {exc}")

    feats = _named(vec, CERT_FEATURE_NAMES)
    available = bool(feats.get("cert_available"))
    reason = None if available else (ssl_error_reason(url, timeout=timeout) or "no certificate observed")
    return SignalBlock(available, reason=reason, features=feats)


async def collect_page_signals(url: str, *, include_page: bool = True) -> SignalBlock:
    """Live JS behaviour, layout geometry and brand-image signals.

    Reuses the page already being rendered for the screenshot modality where
    possible; falls back to its own browser visit when the screenshot is not
    wanted, because the instrumentation has to be installed before navigation.
    """
    from app.preprocessing.page_analysis import (
        BEHAVIOR_FEATURE_NAMES,
        BRANDIMG_FEATURE_NAMES,
        LAYOUT_FEATURE_NAMES,
        extract_behavior_features,
        extract_brand_image_features,
        extract_layout_features,
    )

    if not include_page:
        return SignalBlock(False, reason="page probe not requested")

    from app.services.page_probe import probe_page

    try:
        probe = await probe_page(url)
    except Exception as exc:  # noqa: BLE001
        return SignalBlock(False, reason=f"{exc.__class__.__name__}: {exc}")

    if not probe.available:
        return SignalBlock(False, reason=probe.reason or "page probe unavailable")

    feats = {
        **_named(extract_layout_features(probe.layout), LAYOUT_FEATURE_NAMES),
        **_named(extract_behavior_features(probe.behavior), BEHAVIOR_FEATURE_NAMES),
        **_named(extract_brand_image_features(probe.images), BRANDIMG_FEATURE_NAMES),
    }
    block = SignalBlock(True, features=feats)
    block.features["_n_layout"] = float(len(LAYOUT_FEATURE_NAMES))
    block.features["_n_behavior"] = float(len(BEHAVIOR_FEATURE_NAMES))
    block.features["_n_brandimg"] = float(len(BRANDIMG_FEATURE_NAMES))
    block.features["_elapsed_ms"] = float(probe.elapsed_ms)
    return block


def collect_graph(url: str, *, model_dir: Path | None = None) -> SignalBlock:
    """Domain-node probability from the trained GCN, if a checkpoint exists.

    Blocking; run off the event loop. Returns an unavailable block rather than
    raising when the checkpoint or the graph index is missing, so the API
    degrades cleanly on a fresh clone.
    """
    ckpt = (model_dir or BACKEND_ROOT / "checkpoints") / "graph_model.pt"
    if not ckpt.is_file():
        return SignalBlock(False, reason="graph model not trained")

    try:
        from app.models.graph_model import PhishingGCN
        from app.preprocessing.graph_build import GRAPH_FEATURE_NAMES, build_graph

        import torch

        blob = torch.load(ckpt, map_location="cpu", weights_only=False)
        cfg = dict(blob["model_config"])
        model = PhishingGCN(**cfg)
        model.load_state_dict(blob["model_state"])
        model.eval()

        # One isolated node. Without the full graph there is nothing to pass
        # messages to, so this reduces to the domain's own features through the
        # learned depthwise weights -- a weaker signal than the batch metric in
        # reports/metrics_gnn.md, which does see neighbours.
        spec = build_graph([url], [0], include_shared_tld=False, include_shared_subdomain=False)
        x = torch.tensor(spec.x, dtype=torch.float32)
        if x.shape[1] != int(cfg["in_dim"]):
            return SignalBlock(False, reason="graph feature dimension mismatch")

        mean, std = blob.get("scaler_mean"), blob.get("scaler_std")
        if mean is not None and std is not None:
            x = (x - torch.tensor(mean, dtype=torch.float32)) / torch.tensor(
                std, dtype=torch.float32
            )

        # GraphSpec stores these as numpy; the model wants tensors.
        edge_index = torch.as_tensor(spec.edge_index, dtype=torch.long)
        edge_type = torch.as_tensor(spec.edge_type, dtype=torch.long)

        view = model.inference_view  # bound method; it takes (edge_index, x, edge_type)
        with torch.no_grad():
            prob = view(edge_index, x, edge_type)[0]
        return SignalBlock(True, probability=float(prob))
    except Exception as exc:  # noqa: BLE001
        return SignalBlock(False, reason=f"{exc.__class__.__name__}: {exc}")


async def collect_all(url: str, *, include_page: bool = True) -> dict[str, Any]:
    """Gather every signal group concurrently.

    Certificate probing and the graph pass are blocking socket/tensor work, so
    they are pushed onto a thread rather than run on the event loop.
    """
    cert_task = asyncio.to_thread(collect_certificate, url)
    graph_task = asyncio.to_thread(collect_graph, url)
    page_task = collect_page_signals(url, include_page=include_page)

    cert, graph, page = await asyncio.gather(
        cert_task, graph_task, page_task, return_exceptions=True
    )

    def _block(value: Any, name: str) -> SignalBlock:
        if isinstance(value, SignalBlock):
            return value
        log.warning("signal_failed", signal=name, error=str(value))
        return SignalBlock(False, reason=f"{type(value).__name__}: {value}")

    return {
        "certificate": _block(cert, "certificate"),
        "page": _block(page, "page"),
        "graph": _block(graph, "graph"),
    }


__all__ = ["SignalBlock", "collect_all", "collect_certificate", "collect_graph", "collect_page_signals"]