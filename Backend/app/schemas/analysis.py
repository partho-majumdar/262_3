"""Pydantic request/response models for the detection API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

__all__ = [
    "AnalyzeRequest",
    "ModalityScore",
    "ExplainItem",
    "SignalBlock",
    "AnalyzeResponse",
    "HealthResponse",
]

Modality = Literal["url", "html", "vision"]


class AnalyzeRequest(BaseModel):
    """A URL to analyse.

    ``modalities`` selects which evidence the caller wants fused. This is the
    backend half of the UI's toggle bar: asking for ``["url"]`` runs the
    URL-only path and skips acquisition entirely, so the UI can show each
    modality's standalone verdict on the same input.

    Note the honesty constraint in the docs: a modality that could not be
    acquired is reported as ``available: false`` and is *excluded* from the
    fusion. The system never substitutes a default score for evidence it does
    not have.
    """

    url: str = Field(..., min_length=1, max_length=2048)
    modalities: list[Modality] = Field(
        default_factory=lambda: ["url", "html", "vision"],
        description="Which modalities to acquire and fuse.",
    )
    explain: bool = Field(default=True, description="Compute attribution for each modality.")
    include_signals: bool = Field(
        default=False,
        description=(
            "Collect live signals (TLS certificate, page probe, graph). Off by "
            "default because the certificate probe opens a socket to the target "
            "and the page probe opens a browser, so enabling it adds seconds of "
            "latency to an otherwise fast URL-only verdict. The dashboard turns "
            "it on explicitly."
        ),
    )


class ModalityScore(BaseModel):
    name: Modality
    available: bool
    probability: float | None = Field(
        default=None,
        description="Standalone calibrated probability; null when unavailable.",
    )
    weight: float | None = Field(
        default=None, description="Fusion gate weight for this modality."
    )
    reason: str | None = Field(
        default=None, description="Why the modality is unavailable."
    )
    evidence: dict | None = Field(
        default=None, description="Modality-specific evidence for the UI."
    )


class ExplainItem(BaseModel):
    label: str
    contribution: float


class SignalBlock(BaseModel):
    """One measured signal group.

    ``features`` is reported rather than scored: these come from a live
    observation, and there is no labelled crawl to train a head on, so no
    trained classifier consumes them yet. ``probability`` is only present for
    the graph branch, which does have a real checkpoint behind it.
    """

    available: bool
    reason: str | None = None
    probability: float | None = None
    features: dict[str, float] = Field(default_factory=dict)


class AnalyzeResponse(BaseModel):
    url: str
    verdict: Literal["phishing", "legitimate", "unknown"]
    probability_phishing: float
    confidence: float = Field(
        ...,
        description=(
            "max(p, 1-p) on the served probability. Mathematically redundant with "
            "probability_phishing (it is always >= 0.5) and carries no extra "
            "information; it exists as a convenience display value."
        ),
    )
    probability_is_calibrated: bool = Field(
        ...,
        description=(
            "True when temperature-scaled calibration was applied. False means "
            "the value is a raw sigmoid and should not be read as a frequency."
        ),
    )
    fused: bool = Field(
        ..., description="False when only one modality was available."
    )
    modalities: list[ModalityScore]
    top_features: dict[str, list[ExplainItem]] = Field(default_factory=dict)
    modality_influence: dict = Field(
        default_factory=dict,
        description=(
            "Leave-one-modality-out counterfactual: change in fused phishing "
            "probability when a modality is withheld."
        ),
    )
    signals: dict[str, SignalBlock] = Field(
        default_factory=dict,
        description=(
            "Live-measured signals reported alongside the modelled modalities: "
            "'certificate' (TLS metadata), 'page' (JS behaviour, layout geometry, "
            "brand-image impersonation) and 'graph' (trained GCN probability). "
            "certificate/page features are observed but not yet consumed by a "
            "trained head; graph carries a real model probability."
        ),
    )
    acquisition: dict = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)

    @field_validator("confidence")
    @classmethod
    def _check(cls, v: float) -> float:
        return min(max(v, 0.0), 1.0)


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    url_model: bool
    html_model: bool
    vision_model: bool
    fusion_model: bool
    graph_model: bool = Field(
        default=False,
        description="True when a trained GCN checkpoint exists for domain nodes.",
    )
    screenshot_service: Literal["in_process", "container", "unavailable"]
    notes: list[str] = Field(default_factory=list)
