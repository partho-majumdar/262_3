"""Tests for the shared metrics module.

These check the metrics against analytically known cases. A metrics bug would
silently corrupt every number in the project, so they are tested directly rather
than trusted.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.utils.metrics import (
    TemperatureScaler,
    bootstrap_confidence_interval,
    bootstrap_metric_ci,
    calibration_report,
    compute_binary_metrics,
    platt_scale,
    recall_at_fixed_fpr,
)


def test_confusion_matrix_and_basic_rates():
    # 2 TP, 1 FP, 1 FN, 2 TN
    y = [1, 1, 1, 0, 0, 0]
    p = [0.9, 0.8, 0.2, 0.7, 0.3, 0.1]
    m = compute_binary_metrics(y, p)
    assert m.confusion == {"tp": 2, "tn": 2, "fp": 1, "fn": 1}
    assert m.precision == pytest.approx(2 / 3)
    assert m.recall == pytest.approx(2 / 3)
    assert m.specificity == pytest.approx(2 / 3)
    assert m.accuracy == pytest.approx(4 / 6)


def test_fpr_and_fpr_free_specificity():
    y = [1, 1, 0, 0]
    perfect = compute_binary_metrics(y, [0.99, 0.99, 0.01, 0.01])
    assert perfect.specificity == 1.0
    assert perfect.fpr == 0.0
    assert perfect.recall == 1.0


def test_probability_range_is_respected_in_metrics():
    y = [0, 1, 0, 1]
    m = compute_binary_metrics(y, [0.5, 0.5, 0.5, 0.5])
    for v in (m.accuracy, m.precision, m.recall, m.specificity, m.f1, m.brier):
        assert 0.0 <= v <= 1.0


def test_recall_is_zero_when_everything_is_missed():
    m = compute_binary_metrics([1, 1, 0, 0], [0.1, 0.2, 0.9, 0.8])
    assert m.recall == 0.0
    assert m.confusion["fn"] == 2


def test_mcc_is_plus_one_for_a_perfect_prediction():
    m = compute_binary_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert m.mcc == pytest.approx(1.0)


def test_mcc_is_minus_one_for_always_wrong():
    m = compute_binary_metrics([0, 1], [0.9, 0.1])
    assert m.mcc == pytest.approx(-1.0)


def test_brier_score_matches_definition():
    y = np.array([1, 0, 1, 0])
    p = np.array([0.8, 0.2, 0.6, 0.4])
    expected = float(np.mean((p - y) ** 2))
    assert compute_binary_metrics(y, p).brier == pytest.approx(expected)


def test_auc_is_half_for_a_random_scoring():
    """AUC on coin-flip scores should sit near 0.5, not 1.0."""
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=4000)
    p = rng.random(4000)
    assert compute_binary_metrics(y, p).roc_auc == pytest.approx(0.5, abs=0.05)


def test_auc_is_one_for_a_perfect_ranking():
    y = [0] * 50 + [1] * 50
    p = [0.1] * 50 + [0.9] * 50
    assert compute_binary_metrics(y, p).roc_auc == pytest.approx(1.0)
    assert compute_binary_metrics(y, p).pr_auc == pytest.approx(1.0)


def test_single_class_set_reports_nan_auc_not_a_fake_value():
    """A one-class evaluation set cannot support AUC; NaN is the honest answer."""
    m = compute_binary_metrics([1, 1, 1], [0.5, 0.6, 0.7])
    assert np.isnan(m.roc_auc)
    assert np.isnan(m.pr_auc)


def test_empty_input_is_rejected():
    with pytest.raises(ValueError, match="empty"):
        compute_binary_metrics([], [])


def test_nan_probability_is_rejected():
    with pytest.raises(ValueError, match="NaN"):
        compute_binary_metrics([0, 1], [0.5, np.nan])


def test_recall_at_fixed_fpr_is_monotone_non_increasing():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, size=3000)
    p = np.clip(y * 0.6 + rng.random(3000) * 0.4, 0, 1)
    r = recall_at_fixed_fpr(y, p, (0.01, 0.05, 0.10))
    assert r["0.01"] >= r["0.05"] >= r["0.1"], r


def test_recall_at_fpr_is_nan_for_single_class():
    assert np.isnan(recall_at_fixed_fpr([1, 1], [0.5, 0.6], (0.05,))["0.05"])


def test_calibration_curve_counts_cover_the_sample():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, size=1000)
    p = rng.random(1000)
    m = compute_binary_metrics(y, p, n_calibration_bins=10)
    assert sum(b["count"] for b in m.calibration_bins) == 1000


def test_ece_is_zero_for_a_perfectly_calibrated_set():
    """Every bin holding a single probability value is perfectly calibrated.

    A constant 0.5 prediction on a 50/50 set is calibrated: predicted rate equals
    the empirical rate. (A constant 0.1 on an all-negative set would NOT be,
    which is why the test uses the matching base rate.)
    """
    y = [0] * 50 + [1] * 50
    p = [0.5] * 100
    m = compute_binary_metrics(y, p, n_calibration_bins=10)
    assert m.ece == pytest.approx(0.0, abs=1e-9)


def test_ece_is_large_for_a_confidently_wrong_set():
    """Predicting 0.99 for everything is maximally over-confident at a 50% base rate."""
    y = [0] * 50 + [1] * 50
    p = [0.99] * 100
    assert compute_binary_metrics(y, p, n_calibration_bins=10).ece == pytest.approx(0.49, abs=1e-6)


def _finite_logits(y: np.ndarray, scale: float = 2.0) -> np.ndarray:
    """Logits that separate the classes without producing +/-inf.

    ``log(y/(1-y))`` is infinite for a perfectly separated set, so build the
    logits from a finite probability grid instead.
    """
    p0 = 0.1 + 0.8 * y
    return np.log(p0 / (1.0 - p0)) * scale


def test_bootstrap_interval_brackets_the_point_estimate():
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, size=2000)
    p = np.clip(y * 0.5 + rng.random(2000) * 0.5, 0, 1)
    m = compute_binary_metrics(y, p)
    ci = bootstrap_metric_ci(y, p, "accuracy", n_resamples=200, seed=1)
    assert ci["low"] <= m.accuracy <= ci["high"]


def test_bootstrap_needs_two_samples():
    ci = bootstrap_metric_ci([1], [0.9], "recall")
    assert np.isnan(ci["low"])


def test_bootstrap_confidence_interval_helper():
    values = np.random.default_rng(4).random(500)
    lo, hi = bootstrap_confidence_interval(values, n_resamples=200)
    assert lo < values.mean() < hi


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------
def test_temperature_scaling_is_identity_at_temperature_one():
    ts = TemperatureScaler()
    ts.temperature = 1.0
    logits = np.array([-2.0, 0.0, 3.0])
    assert np.allclose(ts.transform_logits(logits), 1 / (1 + np.exp(-logits)))


def test_temperature_sharpens_under_confident_model():
    ts = TemperatureScaler()
    ts.temperature = 0.5
    logits = np.array([1.0])
    sharpened = ts.transform_logits(logits)[0]
    baseline = 1 / (1 + np.exp(-1.0))
    assert sharpened > baseline


def test_temperature_fit_recovers_a_known_temperature():
    """Temperature scaling must recover the temperature it was generated with.

    Construction (analytic, not guessed): scores z are drawn, the label is
    sampled from sigmoid(z / T_true), but the model reports sigmoid(z). The
    model therefore overstates confidence by a factor of T_true, and fitting must
    recover T_true = 3.

    Note a perfectly separable set cannot test this: MLE drives T toward 0 there,
    which is correct but says nothing about the over-confidence direction.
    """
    rng = np.random.default_rng(0)
    n = 40_000
    z = rng.normal(0.0, 3.0, size=n)
    y = (rng.random(n) < 1.0 / (1.0 + np.exp(-z / 3.0))).astype(int)

    ts = TemperatureScaler().fit(z, y)
    assert 2.5 < ts.temperature < 3.5, f"expected T ~ 3, got {ts.temperature}"

    uncal = compute_binary_metrics(y, 1.0 / (1.0 + np.exp(-z)))
    cal = compute_binary_metrics(y, ts.transform_logits(z))
    assert cal.ece < uncal.ece / 10, f"ECE {uncal.ece} -> {cal.ece}"
    assert cal.brier < uncal.brier


def test_platt_scale_returns_a_usable_sigmoid():
    rng = np.random.default_rng(6)
    y = rng.integers(0, 2, size=2000)
    logits = _finite_logits(y, scale=1.5)
    a, b = platt_scale(logits, y)
    assert a > 0
    assert abs(b) < 1.0


def test_platt_scale_on_single_class_returns_identity():
    assert platt_scale([0.0, 1.0], [1, 1]) == (1.0, 0.0)


def test_calibration_report_compares_all_three_methods():
    rng = np.random.default_rng(7)
    y = rng.integers(0, 2, size=1500)
    logits = _finite_logits(y, scale=2.5)
    rep = calibration_report(y, logits, y, logits)
    for method in ("uncalibrated", "temperature", "platt", "isotonic"):
        assert method in rep["methods"]
        assert 0.0 <= rep["methods"][method]["brier"] <= 1.0
    assert rep["selected_on_validation_brier"] in {"temperature", "platt", "isotonic"}
    assert "max(p_phishing" in rep["confidence_definition"]


def test_calibration_does_not_change_ranking():
    """A monotone transform must leave ROC-AUC untouched."""
    rng = np.random.default_rng(8)
    y = rng.integers(0, 2, size=1000)
    logits = rng.normal(size=1000)
    rep = calibration_report(y, logits, y, logits)
    uncal = rep["methods"]["uncalibrated"]["roc_auc"]
    for method in ("temperature", "platt"):
        assert rep["methods"][method]["roc_auc"] == pytest.approx(uncal, abs=1e-6)