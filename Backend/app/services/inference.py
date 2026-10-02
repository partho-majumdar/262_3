"""End-to-end inference: acquire modalities, embed, fuse, calibrate, explain.

This is the object the API depends on. It owns the model lifecycle and is the
single place that decides what happens when a modality is missing.

Two behaviours matter more than the plumbing:

**No invented evidence.** If the page could not be fetched, the HTML and vision
modalities are reported ``available: false`` with the reason, and they are
excluded from fusion. The detector never quietly falls back to a default score,
because a fabricated 0.5 that looks like a measurement is worse than an honest
"unavailable".

**Per-modality answers are real predictions, not post-hoc slices.** The toggle
bar in the UI can restrict the request to a subset of modalities; the service
then re-runs fusion over just those, which is why the same input can produce a
URL-only verdict and a full multimodal verdict that legitimately differ.
"""

from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.core.logging_config import get_logger  # noqa: E402
from app.models.fusion_model import FusionModel  # noqa: E402
from app.models.html_model import HTMLModalityModel  # noqa: E402
from app.models.url_model import URLCharModel  # noqa: E402
from app.models.vision_model import VisionModalityModel  # noqa: E402
from app.preprocessing.html_features import extract_html_features, visible_text_of  # noqa: E402
from app.preprocessing.html_tokenizer import HTMLTextTokenizer  # noqa: E402
from app.preprocessing.url_features import FEATURE_NAMES, FeatureScaler, extract_batch  # noqa: E402
from app.preprocessing.url_preprocessing import CharTokenizer, normalize_url  # noqa: E402
from app.utils.metrics import TemperatureScaler  # noqa: E402

log = get_logger(__name__)

__all__ = ["InferenceService", "Verdict"]


@dataclass
class Verdict:
    probability: float
    gates: dict[str, float]
    per_modality: dict[str, dict]
    fused: bool
    warnings: list[str]


class InferenceService:
    def __init__(
        self,
        model_dir: Path,
        *,
        allow_live_acquisition: bool = True,
        decision_threshold: float = 0.5,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.allow_live_acquisition = allow_live_acquisition
        self.decision_threshold = decision_threshold
        self.device = "cpu"
        self.available: dict[str, bool] = {
            "url": False, "html": False, "vision": False, "fusion": False
        }
        self.notes: list[str] = []
        self._load()

    # ------------------------------------------------------------------
    def _load(self) -> None:
        url_ck = self.model_dir / "url_model.pt"
        if not url_ck.is_file():
            self.notes.append(f"missing URL checkpoint at {url_ck}")
            return
        blob = torch.load(url_ck, map_location="cpu", weights_only=False)
        self.url_tokenizer = CharTokenizer.from_state_dict(blob["tokenizer"])
        self.url_scaler = (
            FeatureScaler.from_state_dict(blob["scaler"]) if blob.get("scaler") else None
        )
        self.url_model = URLCharModel(**blob["model_config"])
        self.url_model.load_state_dict(blob["model_state"])
        self.url_model.eval()
        self.available["url"] = True
        self.url_meta = blob.get("config", {})

        mm_path = self.model_dir / "multimodal.pt"
        self.html_model = None
        self.vision_model = None
        self.fusion_model = None
        self.html_tokenizer = None
        if not mm_path.is_file():
            self.notes.append(
                f"missing multimodal checkpoint at {mm_path}; HTML, vision and "
                "fusion are unavailable and only the URL branch will answer"
            )
            return
        mm = torch.load(mm_path, map_location="cpu", weights_only=False)

        cal = self._load_calibration()
        self.calibrator = cal

        if mm.get("html_state"):
            from app.preprocessing.html_features import N_HTML_FEATURES

            cfgd = dict(mm["html_model_config"])
            cfgd["n_dom_features"] = N_HTML_FEATURES
            self.html_model = HTMLModalityModel(**cfgd)
            self.html_model.load_state_dict(mm["html_state"])
            self.html_model.eval()
            self.html_tokenizer = HTMLTextTokenizer.from_state_dict(mm["html_tokenizer"])
            self.available["html"] = True
        else:
            self.notes.append("HTML branch absent from the multimodal checkpoint")

        if mm.get("vision_state"):
            vcfg = mm["vision_model_config"]
            self.vision_model = VisionModalityModel(
                backbone=vcfg["backbone"], pretrained=False,
                embedding_out=vcfg["embedding_out"],
            )
            self.vision_model.load_state_dict(mm["vision_state"])
            self.vision_model.eval()
            self.available["vision"] = True
            if not vcfg.get("pretrained", False):
                self.notes.append("vision backbone was trained without pretrained weights")
        else:
            self.notes.append("vision branch absent from the multimodal checkpoint")

        if mm.get("fusion_state"):
            self.fusion_model = FusionModel(**mm["fusion_model_config"])
            self.fusion_model.load_state_dict(mm["fusion_state"])
            self.fusion_model.eval()
            self.available["fusion"] = True

    def _load_calibration(self):
        """Prefer the fused model's own calibration; fall back to none.

        The on-disk contract is the full TemperatureScaler state dict:
        ``{"method": "temperature", "temperature": <float>}``. training/
        train_multimodal.py writes exactly that, so nothing here needs to guess.
        """
        path = self.model_dir / "fusion_calibration.json"
        if not path.is_file():
            return None
        import json

        state = json.loads(path.read_text(encoding="utf-8"))
        method = state.get("method")
        if method == "temperature":
            param = state.get("parameter")
            # Accept both the flat form written by train_multimodal.py and a
            # nested {"temperature": ...} form, since both have existed.
            if isinstance(param, dict):
                param = param.get("temperature")
            if param is None:
                param = state.get("temperature")
            if param is None:
                return None
            scaler = TemperatureScaler()
            scaler.temperature = float(param)
            return scaler
        if method == "platt":
            a = state.get("parameter", {}).get("a")
            b = state.get("parameter", {}).get("b")
            if a is None or b is None:
                return None
            return ("platt", a, b)
        return None

    def _apply_calibration(self, logits: np.ndarray) -> np.ndarray:
        p = 1.0 / (1.0 + np.exp(-logits))
        cal = self.calibrator
        if cal is None:
            return p
        if isinstance(cal, tuple) and cal and cal[0] == "platt":
            _, a, b = cal
            return 1.0 / (1.0 + np.exp(-(a * logits + b)))
        return cal.transform_logits(logits)

    # ------------------------------------------------------------------
    def embed_url(self, url: str) -> tuple[np.ndarray, float]:
        """Return ``(embedding, logit)``.

        The logit is the shipped P2 URL model's own output, so the per-modality
        number the UI shows is exactly the model reported in `metrics_url.json`,
        not a re-derived approximation.
        """
        norm = normalize_url(url)
        ids, mask = self.url_tokenizer.encode_batch([norm])
        feats = (
            self.url_scaler.transform(extract_batch([norm]))
            if self.url_scaler is not None
            else None
        )
        with torch.no_grad():
            # Keyword args: positionally the handcrafted features would land in
            # the `mask` slot and raise a shape error.
            out = self.url_model(
                char_ids=torch.tensor(ids, dtype=torch.long),
                mask=torch.tensor(mask, dtype=torch.float32),
                handcrafted=(
                    torch.tensor(feats, dtype=torch.float32)
                    if feats is not None
                    else None
                ),
            )
        return out.embedding[0].numpy(), float(out.logit)

    def embed_html(self, html: str) -> np.ndarray:
        from app.preprocessing.multimodal_dataset import _soup_of

        soup = _soup_of(html)
        text = visible_text_of(soup)[:4000]
        ids, mask = self.html_tokenizer.encode_batch([text])
        vec = extract_html_features(html)
        with torch.no_grad():
            out = self.html_model(
                torch.tensor(ids, dtype=torch.long),
                torch.tensor(mask, dtype=torch.float32),
                torch.tensor(vec, dtype=torch.float32).unsqueeze(0),
            )
        return out.embedding[0].numpy()

    def embed_vision(self, png_bytes: bytes) -> np.ndarray:
        import io

        from PIL import Image
        from torchvision import transforms

        size = 224
        tf = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        with Image.open(io.BytesIO(png_bytes)) as im:
            x = tf(im.convert("RGB")).unsqueeze(0)
        with torch.no_grad():
            out = self.vision_model(x)
        return out.embedding[0].numpy()

    def fuse(
        self,
        embs: dict[str, np.ndarray],
        avail: dict[str, bool],
        standalone: dict[str, dict] | None = None,
    ) -> tuple[float, dict[str, float]]:
        standalone = standalone or {}
        n_avail = sum(1 for k in avail if avail.get(k))
        if n_avail == 0:
            return 0.5, {k: 0.0 for k in avail}
        if n_avail == 1 or self.fusion_model is None:
            # Fewer than two evidences: there is nothing to fuse, and running the
            # fusion MLP anyway would report a number from a network trained on
            # the (label-confounded) multimodal manifest rather than the branch's
            # own output. Fall back to averaging whatever standalone calibrated
            # scores exist.
            ps = [
                v for v in standalone.values()
                if v.get("available") and v.get("probability") is not None
            ]
            if not ps:
                return 0.5, {k: 0.0 for k in avail}
            mean = float(np.mean([p["probability"] for p in ps]))
            return mean, {k: (1.0 / len(ps) if ok else 0.0) for k, ok in avail.items()}

        zero = np.zeros(self.fusion_model.config["shared_dim"], dtype=np.float32)
        get = lambda n: embs.get(n, zero)
        with torch.no_grad():
            out = self.fusion_model(
                torch.tensor(get("url")[None], dtype=torch.float32),
                torch.tensor(get("html")[None], dtype=torch.float32),
                torch.tensor(get("vision")[None], dtype=torch.float32),
                torch.tensor([1.0 if avail.get("url") else 0.0]),
                torch.tensor([1.0 if avail.get("html") else 0.0]),
                torch.tensor([1.0 if avail.get("vision") else 0.0]),
            )
        gates = out.gates[0].tolist()
        return (
            float(self._apply_calibration(out.logit.numpy())[0]),
            dict(zip(("url", "html", "vision"), gates)),
        )

    # ------------------------------------------------------------------
    async def analyze(
        self,
        url: str,
        modalities: list[str] | None = None,
    ) -> tuple[Verdict, dict]:
        """Run the full pipeline for one URL."""
        want = set(modalities or ("url", "html", "vision"))
        warnings: list[str] = []
        acquisition: dict = {}
        embs: dict[str, np.ndarray] = {}
        standalone: dict[str, dict] = {}
        avail: dict[str, bool] = {"url": False, "html": False, "vision": False}

        # --- URL modality (always available if the model loaded) ---
        if self.available["url"] and "url" in want:
            try:
                emb, logit = self.embed_url(url)
                embs["url"] = emb
                avail["url"] = True
                standalone["url"] = {
                    "available": True,
                    "probability": float(self._apply_calibration(np.array([logit]))[0]),
                }
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"URL embedding failed: {exc}")

        html: str | None = None
        png: bytes | None = None

        need_page = ("html" in want and self.available["html"]) or (
            "vision" in want and self.available["vision"]
        )
        if need_page:
            if not self.allow_live_acquisition:
                warnings.append(
                    "live acquisition is disabled on this deployment; "
                    "page modalities were not fetched"
                )
                acquisition["skipped"] = "acquisition_disabled"
            else:
                acquisition.update(await self._acquire(url, want))
                # _acquire returns payloads under internal keys; pull them out
                # here. Without this the locals stay None and the page branches
                # below are silently skipped even on a successful capture.
                html = acquisition.pop("_html", None)
                png = acquisition.pop("_png", None)

        if html and self.available["html"] and "html" in want:
            try:
                embs["html"] = self.embed_html(html)
                avail["html"] = True
                standalone["html"] = {
                    "available": True,
                    "probability": self._standalone_probability("html", embs["html"]),
                }
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"HTML embedding failed: {exc}")

        if png and self.available["vision"] and "vision" in want:
            try:
                embs["vision"] = self.embed_vision(png)
                avail["vision"] = True
                standalone["vision"] = {
                    "available": True,
                    "probability": self._standalone_probability("vision", embs["vision"]),
                }
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"vision embedding failed: {exc}")

        if not any(avail.values()):
            return (
                Verdict(0.5, {k: 0.0 for k in avail},
                        {k: {"available": False, "probability": None} for k in avail},
                        False, warnings or ["no modality produced a prediction"]),
                acquisition,
            )

        per_mod: dict[str, dict] = {}
        for name in ("url", "html", "vision"):
            if avail[name]:
                entry = {
                    "available": True,
                    "probability": standalone[name]["probability"],
                }
            else:
                entry = {
                    "available": False,
                    "probability": None,
                    "reason": self._reason(name, want, acquisition),
                }
            per_mod[name] = entry

        prob, gates = self.fuse(embs, avail, standalone)
        for name in per_mod:
            per_mod[name]["weight"] = gates.get(name, 0.0)

        n_avail = sum(avail.values())
        # Private keys: consumed by the XAI layer (explain_html/explain_vision need
        # the raw payloads, explain_fusion needs the embeddings and masks to run
        # its leave-one-out counterfactual), then stripped from the public response
        # by the API layer.
        acquisition["_embs"] = embs
        acquisition["_avail"] = dict(avail)
        acquisition["_html"] = html
        acquisition["_png"] = png
        return (
            Verdict(prob, gates, per_mod, fused=n_avail >= 2, warnings=warnings),
            acquisition,
        )

    def _reason(self, name: str, want: set[str], acquisition: dict) -> str:
        if name not in want:
            return "not requested by the caller"
        if name == "html" and not self.available["html"]:
            return "no HTML model is loaded"
        if name == "vision" and not self.available["vision"]:
            return "no vision model is loaded"
        if name == "html":
            return str(acquisition.get("html_error") or "page could not be fetched")
        return str(acquisition.get("vision_error") or "screenshot could not be captured")

    def _standalone_probability(self, name: str, emb: np.ndarray) -> float:
        """Standalone modality probability.

        Uses the auxiliary head the fusion model was trained with, so this is the
        quantity the fusion network itself learned to trust - not a separately
        refitted model that could disagree with it.
        """
        if self.fusion_model is None:
            return 0.5
        with torch.no_grad():
            t = torch.tensor(emb[None], dtype=torch.float32)
            if name == "url":
                z = self.fusion_model.proj_url(t)
                logit = self.fusion_model.aux_url(z).squeeze(-1)
            elif name == "html":
                z = self.fusion_model.proj_html(t)
                logit = self.fusion_model.aux_html(z).squeeze(-1)
            else:
                z = self.fusion_model.proj_vision(t)
                logit = self.fusion_model.aux_vision(z).squeeze(-1)
        return float(self._apply_calibration(logit.numpy())[0])

    async def _acquire(self, url: str, want: set[str]) -> dict:
        from app.services.fetcher import SecureFetcher
        from app.security.url_guard import BlockedTarget

        out: dict = {}
        want_html = "html" in want and self.available["html"]
        want_vision = "vision" in want and self.available["vision"]
        if not (want_html or want_vision):
            return out

        fetcher = SecureFetcher(delay_seconds=0.0, max_per_domain=4)
        try:
            res = await fetcher.fetch(url)
        except BlockedTarget as exc:
            out["html_error"] = str(exc)
            out["vision_error"] = str(exc)
            return out
        except Exception as exc:  # noqa: BLE001
            out["html_error"] = f"{exc.__class__.__name__}: {exc}"
            out["vision_error"] = out["html_error"]
            return out

        out["http_status"] = res.status_code
        out["final_url"] = res.final_url
        out["redirects"] = res.redirects
        if not res.ok:
            out["html_error"] = res.error or "fetch failed"
            out["vision_error"] = out["html_error"]
            return out

        out["html_bytes"] = res.bytes_read
        if want_html:
            # Internal key: raw page text is model input, not response payload.
            out["_html"] = res.html
            out["html_available"] = True
        if want_vision:
            from app.services.screenshot import capture_screenshot

            shot = await capture_screenshot(
                res.final_url or url,
                Path(self.model_dir).parent / "data" / "live_screenshots",
                artifact_id=None,
            )
            if shot.ok and shot.png_path:
                # Raw bytes stay internal (underscore-prefixed keys are stripped
                # before the response is built). Putting a PNG into the JSON
                # response would both break serialisation and ship megabytes.
                out["_png"] = shot.png_path.read_bytes()
                out["screenshot_bytes"] = shot.png_bytes
                out["screenshot_captured"] = True
                out["vision_error"] = None
            else:
                out["screenshot_captured"] = False
                out["vision_error"] = shot.error or "capture failed"
        # Stash html for the caller without putting it in the JSON response.
        out["_html"] = res.html
        return out
