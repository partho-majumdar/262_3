"""Tests for the leakage experiment's plumbing.

The experiment's whole purpose is to be trustworthy about what is and is not
available at inference, so its column selection is tested directly: a feature set
that accidentally includes the label would turn the experiment into a tautology.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))
if str(_BACKEND / "training") not in sys.path:
    sys.path.insert(0, str(_BACKEND / "training"))

# leakage_experiment imports train_url lazily inside main(); load it by path so the
# test works regardless of how pytest was invoked.
_spec = importlib.util.spec_from_file_location(
    "leakage_experiment_under_test", _BACKEND / "training" / "leakage_experiment.py"
)
leakage = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = leakage
_spec.loader.exec_module(leakage)


# ---------------------------------------------------------------------------
# Column selection
# ---------------------------------------------------------------------------
def test_label_is_never_treated_as_a_feature():
    """The source file has its own 'label' column; it is the target, not a signal."""
    assert "label" in leakage.NON_FEATURE_COLUMNS


def test_url_and_row_id_are_not_features():
    assert {"URL", "url", "row_id", "label_raw"} <= leakage.NON_FEATURE_COLUMNS


def test_url_derived_columns_include_the_url_itself():
    assert "URL" in leakage.URL_DERIVED_COLUMNS


def test_page_columns_are_not_classified_as_url_derived():
    """These require a fetched page and must stay in the engineered set."""
    for col in ("LineOfCode", "NoOfCSS", "HasFavicon", "NoOfImage", "Title"):
        assert col not in leakage.URL_DERIVED_COLUMNS


def test_url_derived_and_non_feature_sets_do_not_collide_on_the_label():
    assert "label" not in leakage.URL_DERIVED_COLUMNS


# ---------------------------------------------------------------------------
# Loading engineered columns back onto deduplicated rows
# ---------------------------------------------------------------------------
@pytest.fixture()
def tiny_dataset(tmp_path):
    csv = tmp_path / "dataset.csv"
    pd.DataFrame(
        {
            "URL": ["http://a.test", "http://b.test", "http://a.test", "http://c.test"],
            "URLLength": [12, 12, 12, 12],
            "IsHTTPS": [1, 0, 1, 0],
            "LineOfCode": [10, 20, 10, 30],
            "NoOfCSS": [3, 4, 3, 5],
            "HasFavicon": [1, 0, 1, 0],
            "label": [1, 0, 1, 0],
        }
    ).to_csv(csv, index=False)

    dedup = tmp_path / "dedup.csv"
    pd.DataFrame(
        {
            "row_id": [0, 1, 2],
            "url": ["http://a.test", "http://b.test", "http://c.test"],
            "label_raw": ["1", "0", "0"],
            "registered_domain": ["a.test", "b.test", "c.test"],
        }
    ).to_csv(dedup, index=False)
    return csv, dedup


def test_loader_joins_on_url_and_keeps_every_deduplicated_row(tiny_dataset):
    csv, dedup = tiny_dataset
    out = leakage.load_engineered_for_rows(csv, dedup)
    assert len(out) == 3
    assert set(out["url"]) == {"http://a.test", "http://b.test", "http://c.test"}


def test_loader_keeps_the_page_columns(tiny_dataset):
    csv, dedup = tiny_dataset
    out = leakage.load_engineered_for_rows(csv, dedup)
    assert {"LineOfCode", "NoOfCSS", "HasFavicon"} <= set(out.columns)


def test_loader_takes_the_first_of_a_duplicated_url(tiny_dataset):
    """Matches drop_duplicates(keep='first') used to build dedup.csv."""
    csv, dedup = tiny_dataset
    out = leakage.load_engineered_for_rows(csv, dedup)
    a = out[out["url"] == "http://a.test"].iloc[0]
    assert str(a["LineOfCode"]) == "10"


def test_loader_refuses_an_ambiguous_dedup_file(tiny_dataset):
    csv, dedup = tiny_dataset
    dup = dedup.parent / "dup.csv"
    pd.read_csv(dedup).pipe(lambda d: pd.concat([d, d.iloc[[0]]])).to_csv(dup, index=False)
    with pytest.raises(RuntimeError, match="duplicate URLs"):
        leakage.load_engineered_for_rows(csv, dup)


def test_loader_errors_when_no_rows_match(tiny_dataset):
    csv, dedup = tiny_dataset
    other = dedup.parent / "other.csv"
    # Distinct URLs, none of which occur in the dataset file.
    pd.read_csv(dedup).assign(url=["http://x.test", "http://y.test", "http://z.test"]).to_csv(
        other, index=False
    )
    with pytest.raises(RuntimeError, match="no rows"):
        leakage.load_engineered_for_rows(csv, other)


# ---------------------------------------------------------------------------
# Numeric coercion
# ---------------------------------------------------------------------------
def test_numeric_coercion_handles_non_numeric_cells():
    """An unparseable cell becomes the -1 'missing' sentinel, not NaN."""
    frame = pd.DataFrame({"HasFavicon": ["1", "0"], "LineOfCode": ["10", "abc"]})
    out = leakage._to_numeric(frame, ["HasFavicon", "LineOfCode"])
    assert out.shape == (2, 2)
    assert out[0][1] == 10.0
    assert out[1][1] == -1.0


def test_numeric_coercion_keeps_missing_values_finite():
    """A NaN would propagate into the loss and silently poison training."""
    frame = pd.DataFrame({"LineOfCode": ["10", None]})
    out = leakage._to_numeric(frame, ["LineOfCode"])
    assert out[1][0] == -1.0
    assert pd.notna(out).all()


def test_title_free_text_is_encoded_as_two_numeric_scalars():
    frame = pd.DataFrame({"Title": ["Secure Login", None]})
    out = leakage._to_numeric(frame, ["Title"])
    assert out.shape == (2, 2)
    assert out[0][0] == len("Secure Login")


def test_numeric_coercion_output_is_float32():
    frame = pd.DataFrame({"LineOfCode": ["10", "20"]})
    assert leakage._to_numeric(frame, ["LineOfCode"]).dtype == np.float32


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------
def _payload(**over):
    base = {
        "generated_at_utc": "2026-10-01T00:00:00+00:00",
        "note": (
            "This is a LEAKAGE EXPERIMENT. The engineered columns are computed by "
            "fetching the page, so they are unavailable to a system that receives "
            "only a URL string. Metrics here must not be quoted as achievable system "
            "performance; they quantify what collection-time information is worth."
        ),
        "dataset": {
            "csv_path": "PhiUSIIL.csv",
            "n_rows_joined": 3,
            "n_engineered_columns": 2,
            "engineered_columns": ["LineOfCode", "NoOfCSS"],
            "url_derived_columns_excluded": ["IsHTTPS", "URL"],
            "notable_page_derived_present": ["LineOfCode", "NoOfCSS"],
        },
        "results": {
            "engineered_mlp": {
                "accuracy": 0.9, "precision": 0.9, "recall": 0.9, "specificity": 0.9,
                "f1": 0.9, "roc_auc": 0.95, "pr_auc": 0.94, "mcc": 0.85, "brier": 0.05,
            }
        },
        "comparison_to_url_only_model": {},
        "strongest_single_columns": [{"column": "NoOfCSS", "auc": 0.99}],
    }
    base.update(over)
    return base


def _comparison(delta: float) -> dict:
    side = {
        "accuracy": 0.9, "precision": 0.9, "recall": 0.9, "specificity": 0.9,
        "f1": 0.9, "roc_auc": 0.95, "pr_auc": 0.94, "mcc": 0.85, "brier": 0.05,
    }
    return {
        "url_only_model": side,
        "engineered_mlp": side,
        "delta_recall_mlp_minus_url": delta,
    }


def test_report_is_marked_as_a_leakage_experiment():
    md = leakage._render_markdown(_payload())
    assert "LEAKAGE EXPERIMENT" in md
    assert "must not be quoted as achievable system performance" in md


def test_report_lists_the_excluded_url_columns():
    assert "IsHTTPS" in leakage._render_markdown(_payload())


def test_report_flags_the_corpus_level_similarity_column():
    md = leakage._render_markdown(_payload())
    assert "URLSimilarityIndex" in md
    assert "reference corpus" in md


def test_report_omits_the_comparison_when_url_metrics_are_absent():
    """Before the URL model finishes there is nothing to compare against."""
    assert "higher phishing recall" not in leakage._render_markdown(_payload())


def test_report_states_a_large_engineered_advantage_is_not_deployable():
    md = leakage._render_markdown(
        _payload(comparison_to_url_only_model=_comparison(0.15))
    )
    assert "not deployable" in md


def test_report_states_no_advantage_plainly():
    md = leakage._render_markdown(
        _payload(comparison_to_url_only_model=_comparison(0.001))
    )
    assert "add no usable signal" in md


def test_report_reports_an_engagement_going_the_other_way():
    md = leakage._render_markdown(
        _payload(comparison_to_url_only_model=_comparison(-0.05))
    )
    assert "worse" in md


def test_report_always_says_engineered_columns_are_experiment_only():
    md = leakage._render_markdown(_payload())
    assert "used **only** in this experiment" in md
