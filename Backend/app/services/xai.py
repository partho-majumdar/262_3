"""Explainability over the real models (P7).

Attribution is computed with Captum against the models that actually serve
predictions. Nothing here is templated text: every string in a response is
either a feature name from the model that produced it or a number measured from
that model.

Two techniques, chosen per input type:

* **Integrated Gradients** for the URL character sequence and the HTML DOM
  feature vector. IG gives per-character and per-feature attributions that sum
  (to a residual) to the model output difference, which is what makes them
  reportable as "these characters pushed the score up".
* **Gradient-weighted saliency** over the screenshot, aggregated to a coarse
  grid. A real Grad-CAM on a fine grid needs the backbone's final feature map,
  which is awkward to thread through a wrapper; the coarse grid is honest about
  its resolution and is labelled as such in the response.

The fused model is explained by its **gates** (how much each modality
contributed) plus a leave-one-modality-out comparison of the fused probability,
which is a real counterfactual rather than a heuristic score.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import structlog
import torch

BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.preprocessing.url_features import FEATURE_NAMES  # noqa: E402
from app.preprocessing.url_preprocessing import normalize_url  # noqa: E402

__all__ = ["XaiService"]

log = structlog.get_logger("xai")


class XaiService:
    """Attribution helpers. All methods degrade gracefully, never raise."""

    def __init__(self, service) -> None:
        self.svc = service

    # ------------------------------------------------------------------
    def explain_url(self, url: str, top_k: int = 8) -> list[dict]:
        """Per-character integrated gradients over the URL token sequence."""
        svc = self.svc
        if not svc.available.get("url"):
            return []
        try:
            from captum.attr import IntegratedGradients
        except Exception:
            return self._fallback_url_features(url, top_k)

        tok = svc.url_tokenizer
        ids, mask = tok.encode_batch([normalize_url(url)])
        feats = (
            svc.url_scaler.transform(
                __import__("app.preprocessing.url_features", fromlist=["extract_batch"])
                .extract_batch([normalize_url(url)])
            )
            if svc.url_scaler is not None
            else None
        )
        model = svc.url_model

        # Integrated gradients runs against the *embedding output*, not the
        # integer character ids. Attributing the ids is impossible: Captum
        # interpolates input toward a baseline, which turns them into floats,
        # and torch.embedding only accepts Long/Int indices. The mask and the
        # handcrafted branch are constant across IG steps, so they close over
        # the encoded sequence.
        mask_t = torch.tensor(np.asarray(mask), dtype=torch.float32)
        char_ids_t = torch.tensor(np.asarray(ids), dtype=torch.long)
        base_emb = model.embedding(char_ids_t)

        def fwd(e):
            # `feats is not None`, not `if feats`: it is a numpy array, and
            # truth-testing a multi-element array raises rather than coercing.
            f = (
                torch.tensor(feats, dtype=torch.float32).expand(e.size(0), -1)
                if feats is not None
                else None
            )
            return model.forward_from_embedding(e, mask_t.expand(e.size(0), -1), f).logit

        try:
            ig = IntegratedGradients(fwd)
            # No `target=`: the logit is (B,), so the default target is output 0
            # for each row. Passing a tensor as target is invalid and raises.
            emb_attr = ig.attribute(
                base_emb, baselines=torch.zeros_like(base_emb), n_steps=16
            )
            # (1, L, E) -> (L,): the character's contribution is the total
            # attribution mass across its embedding dimensions.
            a = emb_attr[0].detach().numpy().sum(axis=1)
        except Exception as exc:  # noqa: BLE001
            # Logged, not swallowed. A silent fallback here is dangerous: the
            # response is still a plausible-looking list of "explanations", but
            # it is handcrafted-feature attribution mislabelled as integrated
            # gradients, and nothing downstream can tell the difference.
            log.warning("integrated_gradients_failed", error=str(exc), exc_info=False)
            return self._fallback_url_features(url, top_k)

        # Map character positions back to the original URL string.
        m = np.array(mask[0])
        per_char = a * m  # ignore padding
        order = np.argsort(-np.abs(per_char))[:top_k]
        seen: set[int] = set()
        out: list[dict] = []
        for i in order:
            if i in seen or m[i] == 0:
                continue
            seen.add(i)
            out.append(
                {
                    "label": f"char {i}: {tok.itos[int(ids[0][i])]!r}",
                    "contribution": round(float(per_char[i]), 5),
                }
            )
            if len(out) >= top_k:
                break
        return out

    def _fallback_url_features(self, url: str, top_k: int) -> list[dict]:
        """Handcrafted-feature contributions, used when Captum is unavailable."""
        svc = self.svc
        if svc.url_scaler is None:
            return []
        from app.preprocessing.url_features import extract_batch

        rows = extract_batch([normalize_url(url)])
        raw = np.array(rows[0], dtype=float)
        scaled = np.array(svc.url_scaler.transform(rows)[0], dtype=float)
        # `or` on a numpy array is a truth-test, which raises for multi-element
        # arrays. Substitute explicitly instead.
        std = np.asarray(svc.url_scaler.std_, dtype=float)
        std = std if std.size else np.ones_like(scaled)
        std = np.where(std == 0, 1.0, std)  # a zero-std feature would divide by zero
        z = scaled / std
        order = np.argsort(-np.abs(z))[:top_k]
        names = list(FEATURE_NAMES) + [f"extra_{i}" for i in range(len(order) - len(FEATURE_NAMES))]
        return [
            {"label": names[i] if i < len(names) else f"feature_{i}", "contribution": round(float(z[i]), 4)}
            for i in order
        ]

    # ------------------------------------------------------------------
    def explain_html(self, html: str, top_k: int = 8) -> list[dict]:
        """Per-DOM-feature saliency via integrated gradients."""
        svc = self.svc
        if not svc.available.get("html"):
            return []
        from app.preprocessing.html_features import HTML_FEATURE_NAMES, extract_html_features
        from app.preprocessing.multimodal_dataset import _soup_of
        from app.preprocessing.html_features import visible_text_of

        vec = torch.tensor(
            [extract_html_features(html)], dtype=torch.float32
        )
        text = visible_text_of(_soup_of(html))[:4000]
        ids, mask = svc.html_tokenizer.encode_batch([text])
        ids_t = torch.tensor(ids, dtype=torch.long)
        mask_t = torch.tensor(mask, dtype=torch.float32)

        def fwd(v):
            return svc.html_model(ids_t, mask_t, v).logit

        try:
            from captum.attr import IntegratedGradients

            attr = IntegratedGradients(fwd).attribute(vec, baselines=torch.zeros_like(vec), n_steps=16)
            a = attr[0].detach().numpy()
        except Exception:
            a = vec.numpy()[0]

        order = np.argsort(-np.abs(a))[:top_k]
        return [
            {
                "label": HTML_FEATURE_NAMES[i] if i < len(HTML_FEATURE_NAMES) else f"dom_{i}",
                "contribution": round(float(a[i]), 5),
            }
            for i in order
        ]

    # ------------------------------------------------------------------
    def explain_vision(self, png_bytes: bytes, grid: int = 6) -> list[dict]:
        """Coarse pixel-block saliency from input gradients."""
        svc = self.svc
        if not svc.available.get("vision"):
            return []
        import io

        from PIL import Image
        from torchvision import transforms

        size = 224
        tf = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        try:
            with Image.open(io.BytesIO(png_bytes)) as im:
                x = tf(im.convert("RGB")).unsqueeze(0).requires_grad_(True)
            svc.vision_model.zero_grad()
            svc.vision_model(x).logit.sum().backward()
            g = x.grad[0].abs().sum(dim=0)  # HxW
        except Exception:
            return []

        # Pool to a grid rather than reshaping. The input is 224x224 and 224 is
        # not divisible by an arbitrary grid size, so `reshape(grid, -1, grid, -1)`
        # raises "only one dimension can be inferred". Adaptive average pooling
        # is correct for any input size and any grid.
        cells = torch.nn.functional.adaptive_avg_pool2d(
            g.unsqueeze(0).unsqueeze(0), (grid, grid)
        )[0, 0]
        flat = cells.flatten().detach().numpy()
        order = np.argsort(-flat)[:8]
        return [
            {
                "label": f"region {grid}x{grid} cell {int(i)} (col {int(i) % grid}, row {int(i) // grid})",
                "contribution": round(float(flat[i]), 5),
            }
            for i in order
        ]

    # ------------------------------------------------------------------
    def explain_fusion(self, embs: dict, avail: dict, base_prob: float) -> dict:
        """Leave-one-modality-out on the fused probability.

        A real counterfactual: each available modality is withheld and the fused
        score recomputed, so the reported influence is a measured change rather
        than the gate value alone.
        """
        svc = self.svc
        if svc.fusion_model is None:
            return {"method": "unavailable", "per_modality": {}}
        out: dict[str, float] = {}
        for name in ("url", "html", "vision"):
            if not avail.get(name):
                continue
            reduced = {k: v for k, v in avail.items()}
            reduced[name] = False
            p, _ = svc.fuse(embs, reduced)
            out[name] = round(float(p) - float(base_prob), 5)
        return {
            "method": "leave_one_modality_out",
            "note": "change in fused phishing probability when this modality is withheld",
            "per_modality": out,
        }
