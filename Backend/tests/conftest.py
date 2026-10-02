"""Shared pytest fixtures.

The split tests read the *real* split files produced by
``training/make_splits.py``. If they are missing the test module skips with an
explicit message rather than fabricating a split, because a test that silently
builds its own fixture would not be testing the artefacts we ship.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

SPLITS_DIR = BACKEND_ROOT / "data" / "splits"
REPORTS_DIR = BACKEND_ROOT / "reports"


@pytest.fixture(scope="session")
def splits_dir() -> Path:
    """Location of the real split artefacts."""
    return SPLITS_DIR


@pytest.fixture(scope="session")
def dedup_csv(splits_dir: Path) -> Path:
    path = splits_dir / "dedup.csv"
    if not path.is_file():
        pytest.skip(
            f"Missing {path}. Run: python training/make_splits.py --csv <PhiUSIIL.csv> "
            "to produce the real split before running this test."
        )
    return path


@pytest.fixture(scope="session")
def domain_split_csv(splits_dir: Path) -> Path:
    path = splits_dir / "domain_split.csv"
    if not path.is_file():
        pytest.skip(
            f"Missing {path}. Run: python training/make_splits.py --csv <PhiUSIIL.csv> first."
        )
    return path


@pytest.fixture(scope="session")
def split_summary_json(splits_dir: Path) -> Path:
    path = splits_dir / "split_summary.json"
    if not path.is_file():
        pytest.skip(f"Missing {path}. Run training/make_splits.py first.")
    return path


@pytest.fixture(scope="session")
def dedup_df(dedup_csv: Path):
    """The deduplicated URL table the split indices refer to."""
    import pandas as pd

    return pd.read_csv(dedup_csv, dtype={"url": str, "label_raw": str})


@pytest.fixture(scope="session")
def split_indices(splits_dir: Path) -> dict[str, list[int]]:
    """Parsed ``{split: [row_id, ...]}`` from the real index files."""
    import json

    out: dict[str, list[int]] = {}
    for name in ("train", "val", "test"):
        path = splits_dir / f"{name}.txt"
        if not path.is_file():
            pytest.skip(f"Missing {path}. Run training/make_splits.py first.")
        ids = [int(line) for line in path.read_text(encoding="utf-8").split() if line.strip()]
        out[name] = ids
    return out
