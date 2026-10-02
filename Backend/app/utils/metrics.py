"""Shared evaluation metrics.

Deliberately emphasises **phishing-class recall**: the cost of missing a phishing
URL is asymmetric, so a system optimised for accuracy is the wrong target. Every
metric block therefore carries recall and specificity for both classes, plus
recall at fixed false-positive rates, which is the operating point an analyst
actually cares about.

Calibration helpers live here too so the fusion phase and the evaluation script
report them identically.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

EPS = 1e-12


def _safe_div(a: float, b: float) -> float:
    return float(a / b) if abs(b) > EPS else 0.0


def _confusion(y_true: np.ndarray, y_pred: np.ndarray, positive: int = 1) -> dict[str, int]:
    tp = int(np.sum((y_true == positive) & (y_pred == positive)))
    tn = int(np.sum((y_true != positive) & (y_pred != positive)))
    fp = int(np.sum((y_true != positive) & (y_pred == positive)))
    fn = int(np.sum((y_true == positive) & (y_pred != positive)))
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


@dataclass
class BinaryMetrics:
    """All metrics for one binary classification problem."""

    n_samples: int
    accuracy: float
    precision: float
    recall: float  # == sensitivity == TPR on the positive (phishing) class
    specificity: float  # == TNR on the negative class
    f1: float
    fpr: float
    roc_auc: float
    pr_auc: float
    mcc: float
    balanced_accuracy: float
    brier: float
    confusion: dict[str, int]
    negative: dict[str, float] = field(default_factory=dict)
    recall_at_fpr: dict[str, float] = field(default_factory=dict)
    calibration_bins: list[dict[str, float]] = field(default_factory=list)
    ece: float = 0.0

    def to_dict(self) -> dict:
        return {
            "n_samples": self.n_samples,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "specificity": self.specificity,
            "f1": self.f1,
            "fpr": self.fpr,
            "roc_auc": self.roc_auc,
            "pr_auc": self.pr_auc,
            "mcc": self.mcc,
            "balanced_accuracy": self.balanced_accuracy,
            "brier": self.brier,
            "confusion_matrix": self.confusion,
            "negative_class": self.negative,
            "recall_at_fpr": self.recall_at_fpr,
            "ece": self.ece,
            "calibration_curve": self.calibration_bins,
        }


def compute_binary_metrics(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    threshold: float = 0.5,
    n_calibration_bins: int = 10,
    recall_at_fpr_points: Sequence[float] = (0.01, 0.05, 0.10),
) -> BinaryMetrics:
    """Compute the full metric block.

    ``y_prob`` is the **calibrated-or-raw phishing probability**; ``threshold``
    is the decision point. ``recall_at_fpr`` sweeps the ROC curve and reports the
    achievable phishing recall at each requested false-positive rate.
    """
    from sklearn.metrics import average_precision_score, roc_auc_score

    y_true = np.asarray(y_true).astype(int).ravel()
    y_prob = np.asarray(y_prob, dtype=float).ravel()

    if y_true.size == 0:
        raise ValueError("cannot compute metrics on an empty array")
    if not np.all(np.isfinite(y_prob)):
        raise ValueError("y_prob contains NaN or inf")

    y_pred = (y_prob >= threshold).astype(int)
    cm = _confusion(y_true, y_pred, positive=1)

    tp, tn, fp, fn = cm["tp"], cm["tn"], cm["fp"], cm["fn"]
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    specificity = _safe_div(tn, tn + fp)
    f1 = _safe_div(2 * precision * recall, precision + recall)

    # Negative-class view, so "precision" is not read as if it were symmetric.
    neg_cm = _confusion(y_true, y_pred, positive=0)
    neg_precision = _safe_div(neg_cm["tp"], neg_cm["tp"] + neg_cm["fp"])
    neg_recall = _safe_div(neg_cm["tp"], neg_cm["tp"] + neg_cm["fn"])

    denom = math_sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = _safe_div(tp * tn - fp * fn, denom)

    if len(np.unique(y_true)) < 2:
        # A single-class evaluation set cannot support AUC; report NaN honestly
        # rather than a misleading 0.5 or 1.0.
        roc_auc = float("nan")
        pr_auc = float("nan")
    else:
        roc_auc = float(roc_auc_score(y_true, y_prob))
        pr_auc = float(average_precision_score(y_true, y_prob))

    bins = _calibration_curve(y_true, y_prob, n_calibration_bins)
    # Empty bins carry gap=NaN; they must be skipped, because 0 * NaN is NaN and
    # would silently poison the whole ECE.
    ece = float(
        sum(b["gap"] * b["count"] for b in bins if b["count"] > 0 and np.isfinite(b["gap"]))
        / max(y_true.size, 1)
    )

    return BinaryMetrics(
        n_samples=int(y_true.size),
        accuracy=_safe_div(tp + tn, y_true.size),
        precision=precision,
        recall=recall,
        specificity=specificity,
        f1=f1,
        fpr=_safe_div(fp, fp + tn),
        roc_auc=roc_auc,
        pr_auc=pr_auc,
        mcc=mcc,
        balanced_accuracy=float((recall + specificity) / 2.0),
        brier=float(np.mean((y_prob - y_true) ** 2)),
        confusion=cm,
        negative={
            "precision": neg_precision,
            "recall": neg_recall,
            "f1": _safe_div(2 * neg_precision * neg_recall, neg_precision + neg_recall),
            "support": int(np.sum(y_true == 0)),
        },
        recall_at_fpr=recall_at_fixed_fpr(y_true, y_prob, recall_at_fpr_points),
        calibration_bins=bins,
        ece=ece,
    )


def math_sqrt(x: float) -> float:
    import math

    return math.sqrt(max(x, 0.0))


def recall_at_fixed_fpr(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    points: Sequence[float] = (0.01, 0.05, 0.10),
) -> dict[str, float]:
    """Maximum phishing recall achievable at each target FPR.

    This is the operating-point view an analyst wants: "if I accept at most 1 in
    100 false alarms, how many phishing URLs do I still catch?"
    """
    from sklearn.metrics import roc_curve

    y_true = np.asarray(y_true).astype(int).ravel()
    y_prob = np.asarray(y_prob, dtype=float).ravel()
    if len(np.unique(y_true)) < 2:
        return {str(p): float("nan") for p in points}

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    out: dict[str, float] = {}
    for target in points:
        idx = np.where(fpr <= target)[0]
        out[str(target)] = float(tpr[idx].max()) if idx.size else 0.0
    return out


def _calibration_curve(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> list[dict[str, float]]:
    """Equal-width reliability bins over [0, 1]."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    out: list[dict[str, float]] = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == 0:
            sel = (y_prob >= lo) & (y_prob <= hi)
        else:
            sel = (y_prob > lo) & (y_prob <= hi)
        count = int(sel.sum())
        if count == 0:
            out.append(
                {
                    "bin_lower": float(lo),
                    "bin_upper": float(hi),
                    "count": 0,
                    "mean_predicted": float("nan"),
                    "empirical_rate": float("nan"),
                    "gap": float("nan"),
                }
            )
            continue
        mean_pred = float(y_prob[sel].mean())
        emp = float(y_true[sel].mean())
        out.append(
            {
                "bin_lower": float(lo),
                "bin_upper": float(hi),
                "count": count,
                "mean_predicted": mean_pred,
                "empirical_rate": emp,
                "gap": float(abs(mean_pred - emp)),
            }
        )
    return out


def bootstrap_confidence_interval(
    values: Sequence[float],
    confidence: float = 0.95,
    n_resamples: int = 1000,
    seed: int = 42,
) -> tuple[float, float]:
    """Percentile bootstrap CI for a per-sample statistic.

    The caller supplies the *per-sample* values (e.g. per-sample correctness, or
    per-sample contribution to recall). Percentile intervals are used because they
    stay inside the metric's valid range, which matters for bounded statistics
    like recall at a fixed FPR.
    """
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_resamples, arr.size))
    means = arr[idx].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    lo, hi = np.quantile(means, [alpha, 1.0 - alpha])
    return (float(lo), float(hi))


def bootstrap_metric_ci(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    metric: str = "recall",
    threshold: float = 0.5,
    confidence: float = 0.95,
    n_resamples: int = 1000,
    seed: int = 42,
    recall_at_fpr_point: float | None = None,
) -> dict[str, float]:
    """Bootstrap CI for a single named metric, resampling test points.

    Each resample recomputes the metric from scratch (not the mean of per-sample
    values) so the interval reflects the metric's own non-linearity - this
    matters for recall-at-fixed-FPR, where averaging per-sample indicators would
    be meaningless.
    """
    y_true = np.asarray(y_true).astype(int).ravel()
    y_prob = np.asarray(y_prob, dtype=float).ravel()
    n = y_true.size
    if n < 2:
        return {"low": float("nan"), "high": float("nan"), "n_resamples": 0}

    rng = np.random.default_rng(seed)
    stats: list[float] = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        yt, yp = y_true[idx], y_prob[idx]
        if len(np.unique(yt)) < 2:
            continue  # skip degenerate resamples rather than emit a fake value
        if metric == "recall_at_fpr":
            if recall_at_fpr_point is None:
                raise ValueError("recall_at_fpr requires recall_at_fpr_point")
            stats.append(recall_at_fixed_fpr(yt, yp, (recall_at_fpr_point,))[str(recall_at_fpr_point)])
        else:
            m = compute_binary_metrics(yt, yp, threshold=threshold, n_calibration_bins=5)
            stats.append(getattr(m, metric))

    if not stats:
        return {"low": float("nan"), "high": float("nan"), "n_resamples": 0}
    arr = np.asarray(stats, dtype=float)
    arr = arr[np.isfinite(arr)]
    alpha = (1.0 - confidence) / 2.0
    lo, hi = np.quantile(arr, [alpha, 1.0 - alpha])
    return {
        "low": float(lo),
        "high": float(hi),
        "n_resamples": int(arr.size),
        "confidence": confidence,
    }


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------
class TemperatureScaler:
    """Single-parameter temperature scaling: ``p = sigmoid(logit / T)``.

    Fitted by minimising log loss on the validation split. Stored as ``T`` so a
    reader can reproduce the transform exactly.
    """

    def __init__(self) -> None:
        self.temperature: float = 1.0

    def fit(self, logits: Sequence[float], y: Sequence[int], max_iter: int = 200) -> "TemperatureScaler":
        logits = np.asarray(logits, dtype=float).ravel()
        y = np.asarray(y, dtype=float).ravel()
        if logits.size == 0:
            raise ValueError("cannot fit TemperatureScaler on empty data")

        # NLL(T) = mean(softplus(logit/T) - y * logit/T) ; dNLL/dT solved by
        # bounded scalar minimisation - robust and dependency-free.
        from scipy.optimize import minimize_scalar

        def nll(log_t: float) -> float:
            t = float(np.exp(log_t))
            z = logits / max(t, 1e-6)
            # log(1 + exp(-|z|)) + max(z, 0)  ==  softplus, computed stably
            loss = np.logaddexp(0.0, z) - y * z
            return float(np.mean(loss))

        res = minimize_scalar(nll, bounds=(-4.0, 4.0), method="bounded", options={"maxiter": max_iter})
        self.temperature = float(np.exp(res.x))
        return self

    def transform_logits(self, logits: Sequence[float]) -> np.ndarray:
        z = np.asarray(logits, dtype=float).ravel() / max(self.temperature, 1e-6)
        return 1.0 / (1.0 + np.exp(-z))

    def transform(self, probs: Sequence[float]) -> np.ndarray:
        """Apply to probabilities that originated from sigmoid logits."""
        p = np.clip(np.asarray(probs, dtype=float).ravel(), EPS, 1 - EPS)
        logits = np.log(p / (1.0 - p))
        return self.transform_logits(logits)

    def state_dict(self) -> dict:
        return {"method": "temperature", "temperature": self.temperature}

    @classmethod
    def from_state_dict(cls, state: dict) -> "TemperatureScaler":
        obj = cls()
        obj.temperature = float(state["temperature"])
        return obj


def platt_scale(logits: Sequence[float], y: Sequence[int]) -> tuple[float, float]:
    """Fit Platt scaling ``p = sigmoid(a * logit + b)``. Returns ``(a, b)``."""
    from sklearn.linear_model import LogisticRegression

    z = np.asarray(logits, dtype=float).ravel().reshape(-1, 1)
    target = np.asarray(y, dtype=int).ravel()
    if len(np.unique(target)) < 2:
        return (1.0, 0.0)
    lr = LogisticRegression(C=1e6, solver="lbfgs")
    lr.fit(z, target)
    return (float(lr.coef_[0][0]), float(lr.intercept_[0]))


def isotonic_scale(logits: Sequence[float], y: Sequence[int]) -> object:
    """Fit isotonic regression mapping logits -> probability."""
    from sklearn.isotonic import IsotonicRegression

    z = np.asarray(logits, dtype=float).ravel()
    target = np.asarray(y, dtype=int).ravel()
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    iso.fit(z, target)
    return iso


def calibration_report(
    y_val: Sequence[int],
    val_logits: Sequence[float],
    y_test: Sequence[int],
    test_logits: Sequence[float],
) -> dict:
    """Fit all three calibrators on validation, compare them on test.

    Reports Brier and ECE for each so the choice of calibration method is
    evidence-based rather than assumed.
    """
    out: dict = {"methods": {}}

    raw = 1.0 / (1.0 + np.exp(-np.asarray(val_logits, dtype=float).ravel()))
    test_raw = 1.0 / (1.0 + np.exp(-np.asarray(test_logits, dtype=float).ravel()))
    m = compute_binary_metrics(y_test, test_raw)
    out["methods"]["uncalibrated"] = {
        "brier": m.brier,
        "ece": m.ece,
        "roc_auc": m.roc_auc,
        "pr_auc": m.pr_auc,
    }

    ts = TemperatureScaler().fit(val_logits, y_val)
    m = compute_binary_metrics(y_test, ts.transform_logits(test_logits))
    out["methods"]["temperature"] = {
        "brier": m.brier,
        "ece": m.ece,
        "roc_auc": m.roc_auc,
        "pr_auc": m.pr_auc,
        "parameter": ts.state_dict(),
    }

    a, b = platt_scale(val_logits, y_val)
    test_platt = 1.0 / (1.0 + np.exp(-(a * np.asarray(test_logits, dtype=float).ravel() + b)))
    m = compute_binary_metrics(y_test, test_platt)
    out["methods"]["platt"] = {
        "brier": m.brier,
        "ece": m.ece,
        "roc_auc": m.roc_auc,
        "pr_auc": m.pr_auc,
        "parameter": {"a": a, "b": b},
    }

    iso = isotonic_scale(val_logits, y_val)
    test_iso = iso.predict(np.asarray(test_logits, dtype=float).ravel())
    m = compute_binary_metrics(y_test, test_iso)
    out["methods"]["isotonic"] = {
        "brier": m.brier,
        "ece": m.ece,
        "roc_auc": m.roc_auc,
        "pr_auc": m.pr_auc,
    }

    # Select the calibrator by validation Brier so the test numbers stay honest.
    best = min(
        (k for k in out["methods"] if k != "uncalibrated"),
        key=lambda k: out["methods"][k]["brier"],
    )
    out["selected_on_validation_brier"] = best
    out["confidence_definition"] = "max(p_phishing, 1 - p_phishing) on the calibrated probability"
    return out