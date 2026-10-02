"""FastAPI application (P8).

Deliberately **no authentication**. This is a research tool and the brief is
explicit that no login is wanted. That choice carries a real consequence which
is stated in ``/health`` and in the docs rather than left implicit: with no
auth, **anyone who can reach this service can make it fetch arbitrary URLs on
the server's network**. The SSRF guard in :mod:`app.security.url_guard` is
therefore the security boundary here, not authentication. Do not expose this
service to an untrusted network without putting a reverse proxy in front of it.
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.core.config import get_settings  # noqa: E402
from app.core.logging_config import configure_logging, get_logger  # noqa: E402
from app.schemas.analysis import (  # noqa: E402
    AnalyzeRequest,
    AnalyzeResponse,
    ExplainItem,
    HealthResponse,
    ModalityScore,
)
from app.security.url_guard import BlockedTarget  # noqa: E402
from app.services.inference import InferenceService  # noqa: E402
from app.services.xai import XaiService  # noqa: E402

configure_logging(fmt="json")
log = get_logger(__name__)

app = FastAPI(
    title="Multimodal Phishing URL Detector",
    version="1.0.0",
    description=(
        "Fuses the raw URL string, the fetched HTML, and a rendered screenshot "
        "into one phishing probability. No authentication. Calibration is applied "
        "only when checkpoints/fusion_calibration.json is present; /health reports "
        "whether it is active."
    ),
)

# Dev convenience only: the frontend runs on a different port in development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

_settings = get_settings()
_settings.ensure_dirs()

_service: InferenceService | None = None
_xai: XaiService | None = None

#: Fixed-window rate limit. Bounds how fast this service can be used to make the
#: server fetch things, which is the main abuse vector when auth is absent.
_RATE: dict[str, list[float]] = {}
_RATE_LIMIT = _settings.rate_limit_per_minute


def service() -> InferenceService:
    global _service, _xai
    if _service is None:
        _service = InferenceService(_settings.model_dir)
        _xai = XaiService(_service)
        log.info("service_loaded", available=_service.available, notes=_service.notes)
    return _service


def xai() -> XaiService:
    service()
    assert _xai is not None
    return _xai


def _rate_limit(client: str) -> None:
    now = time.time()
    hits = [t for t in _RATE.get(client, []) if now - t < 60.0]
    if len(hits) >= _RATE_LIMIT:
        raise HTTPException(
            status_code=429,
            detail=f"rate limit exceeded ({_RATE_LIMIT} requests/minute); try again shortly",
        )
    hits.append(now)
    _RATE[client] = hits


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    if request.url.path == "/analyze":
        client = request.client.host if request.client else "unknown"
        try:
            _rate_limit(client)
        except HTTPException as exc:
            from fastapi.responses import JSONResponse

            return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
    return await call_next(request)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    svc = service()
    missing = [n for n, ok in svc.available.items() if not ok]
    graph_ckpt = Path(_settings.model_dir) / "graph_model.pt"
    return HealthResponse(
        status="ok" if all(svc.available.values()) else "degraded",
        url_model=svc.available["url"],
        html_model=svc.available["html"],
        vision_model=svc.available["vision"],
        fusion_model=svc.available["fusion"],
        graph_model=graph_ckpt.is_file(),
        # Docker is unavailable on this host, so capture runs in-process with the
        # browser hardened from the inside rather than inside a container.
        screenshot_service="in_process",
        notes=svc.notes
        + [
            "No authentication: anyone who can reach this service can make it "
            "fetch URLs. Do not expose it to an untrusted network without a "
            "reverse proxy in front.",
            (
                "Probability calibration is active (temperature scaling fitted on "
                "the validation split)."
                if svc.calibrator is not None
                else "Probability calibration is NOT active: checkpoints/"
                "fusion_calibration.json is absent, so the served probability is "
                "a raw sigmoid of the fusion logit."
            ),
        ]
        + ([f"unavailable components: {', '.join(missing)}"] if missing else []),
    )


@app.get("/metrics")
def metrics() -> dict:
    """The measured numbers behind the served models, for the UI to display."""
    import json

    out: dict = {}
    reports = _settings.reports_dir
    for name, fname in (
        ("url_model", "metrics_url.json"),
        ("multimodal", "metrics_multimodal.json"),
    ):
        p = reports / fname
        if p.is_file():
            data = json.loads(p.read_text(encoding="utf-8"))
            if name == "url_model":
                out[name] = {
                    "test_metrics_calibrated": data.get("test_metrics_calibrated"),
                    "bootstrap_ci_95": data.get("bootstrap_ci_95"),
                    "note": "trained on 40,000 URL-only rows (dev profile)",
                }
            else:
                out[name] = {
                    "html": (data.get("html") or {}).get("test"),
                    "vision": (data.get("vision") or {}).get("test"),
                    "fusion": (data.get("fusion") or {}).get("test"),
                    "rows": data.get("rows"),
                    "note": (
                        "trained on the pages still reachable in 2026 from a 2022 "
                        "crawl; not comparable to the URL-only figures"
                    ),
                }
    return out


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(req: AnalyzeRequest) -> AnalyzeResponse:
    svc = service()

    # Validate before doing anything expensive. The same guard the fetcher uses,
    # applied up front so a blocked URL is rejected without a network round trip.
    # DNS is blocking, so it goes to a thread and off the event loop.
    try:
        from app.security.url_guard import validate_target

        target = await asyncio.to_thread(
            validate_target,
            req.url,
            allowed_schemes=_settings.allowed_schemes,
            allowed_ports=_settings.allowed_ports,
            max_length=_settings.max_url_length,
            # An unresolvable host is a dead domain, not an SSRF risk. Still
            # analyse the URL string: a taken-down or short-lived phishing domain
            # is exactly the case a detector should not be blind to.
            allow_unresolved=True,
        )
    except BlockedTarget as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    verdict, acquisition = await svc.analyze(req.url, req.modalities)
    if not target.resolved and target.dns_error:
        # Surfaced rather than hidden, and not as a warning: the page simply
        # could not be fetched, so the html and vision modalities are absent.
        acquisition["dns_resolved"] = False
        acquisition["dns_error"] = target.dns_error

    if verdict.probability >= 0.5:
        label = "phishing"
    elif verdict.probability > 0.0:
        label = "legitimate"
    else:
        label = "unknown"

    modalities = [
        ModalityScore(
            name=name,  # type: ignore[arg-type]
            available=bool(info["available"]),
            probability=info.get("probability"),
            weight=info.get("weight"),
            reason=info.get("reason"),
        )
        for name, info in verdict.per_modality.items()
    ]

    # Payloads the XAI layer needs before the private keys are stripped below.
    xai_html = acquisition.pop("_html", None)
    xai_png = acquisition.pop("_png", None)
    xai_embs = acquisition.pop("_embs", None)
    xai_avail = acquisition.pop("_avail", None)

    top_features: dict[str, list[ExplainItem]] = {}
    fusion_explanation: dict | None = None
    if req.explain:
        try:
            ex = xai()
            top_features["url"] = [
                ExplainItem(**e) for e in ex.explain_url(req.url)
            ]
        except Exception as exc:  # noqa: BLE001
            log.warning("xai_url_failed", error=str(exc))

        if xai_html:
            try:
                top_features["html"] = [
                    ExplainItem(**e) for e in ex.explain_html(xai_html)
                ]
            except Exception as exc:  # noqa: BLE001
                log.warning("xai_html_failed", error=str(exc))

        if xai_png:
            try:
                top_features["vision"] = [
                    ExplainItem(**e) for e in ex.explain_vision(xai_png)
                ]
            except Exception as exc:  # noqa: BLE001
                log.warning("xai_vision_failed", error=str(exc))

        if xai_embs and xai_avail:
            try:
                fusion_explanation = ex.explain_fusion(
                    xai_embs, xai_avail, float(verdict.probability)
                )
            except Exception as exc:  # noqa: BLE001
                log.warning("xai_fusion_failed", error=str(exc))

    pub = {k: v for k, v in acquisition.items() if not k.startswith("_")}

    # Live signals: TLS certificate, page probe (JS behaviour / layout / brand
    # images) and the graph branch. Collected after the verdict so a slow probe
    # never delays the core answer, and gathered concurrently because each one
    # is network- or CPU-bound. Every collector degrades to an unavailable block
    # with a reason rather than raising.
    signals: dict = {}
    if req.include_signals:
        try:
            from app.services.signals import collect_all

            raw_signals = await collect_all(
                req.url, include_page=bool({"html", "vision"} & set(req.modalities))
            )
            signals = {k: v.to_dict() for k, v in raw_signals.items()}
        except Exception as exc:  # noqa: BLE001
            log.warning("signals_failed", error=str(exc))
            signals = {}

    return AnalyzeResponse(
        url=req.url,
        verdict=label,  # type: ignore[arg-type]
        probability_phishing=round(float(verdict.probability), 6),
        confidence=round(max(verdict.probability, 1 - verdict.probability), 6),
        probability_is_calibrated=service().calibrator is not None,
        fused=verdict.fused,
        modalities=modalities,
        top_features=top_features,
        modality_influence=fusion_explanation or {},
        signals=signals,
        acquisition=pub,
        warnings=verdict.warnings,
    )
