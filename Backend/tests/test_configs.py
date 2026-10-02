"""Config and path-resolution tests.

Both training configs must parse, and a relative ``--config`` must resolve to
paths inside ``backend/``. A silent mis-resolution would send the trainer
looking for a splits directory that does not exist, or - worse - silently read
the wrong data.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from training.train_url import load_config, resolve_paths

BACKEND_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = BACKEND_ROOT / "configs"


@pytest.mark.parametrize("name", ["dev.yaml", "full.yaml"])
def test_config_parses(name: str):
    """Regression: an unquoted value containing ': ' is invalid YAML."""
    cfg = load_config(CONFIG_DIR / name)
    assert cfg["profile"] in {"dev", "full"}
    assert "url_model" in cfg and "fusion" in cfg


def test_dev_and_full_have_identical_architecture():
    """Structural knobs must match across profiles; only budget knobs may differ.

    ``structural`` = the shape of the model/system. Changing one of these makes
    the two profiles incomparable, so it must be applied to both.

    ``budget`` = capacity and schedule. These are explicitly allowed to differ
    (documented in docs/architecture.md) - epochs, batch size, channel widths,
    layer counts, row caps, bootstrap count, and dropout rates.
    """
    dev = load_config(CONFIG_DIR / "dev.yaml")
    full = load_config(CONFIG_DIR / "full.yaml")

    structural = [
        ("url_model", "cnn_kernel_sizes"),
        ("url_model", "bidirectional"),
        ("url_model", "embedding_out"),
        ("url_model", "handcrafted_branch"),
        ("url_model", "class_weighting"),
        ("html_model", "embedding_out"),
        ("fusion", "shared_dim"),
        ("fusion", "min_available_modalities"),
        ("fusion", "baselines"),
        ("xai", "url_method"),
        ("xai", "html_method"),
        ("xai", "fusion_method"),
        ("split", "grouping"),
    ]
    for section, key in structural:
        assert dev[section][key] == full[section][key], (
            f"{section}.{key} differs between dev and full "
            f"({dev[section][key]!r} vs {full[section][key]!r}); that is a structural "
            "change and must be applied to both profiles"
        )


def test_dropout_rates_are_classified_as_budget_not_structure():
    """Guards the classification: dropout may differ, it is not architecture."""
    dev = load_config(CONFIG_DIR / "dev.yaml")
    full = load_config(CONFIG_DIR / "full.yaml")
    # Documented budget knobs that legitimately differ between profiles.
    assert dev["fusion"]["modality_dropout"] != full["fusion"]["modality_dropout"]
    assert dev["url_model"]["dropout"] != full["url_model"]["dropout"]
    # ...but the vision transform must not differ, since that changes the input
    # pipeline rather than just its cost.
    assert dev["vision_model"]["augment"] == full["vision_model"]["augment"]


def test_dev_profile_is_smaller_than_full():
    dev = load_config(CONFIG_DIR / "dev.yaml")
    full = load_config(CONFIG_DIR / "full.yaml")
    assert dev["url_model"]["epochs"] < full["url_model"]["epochs"]
    assert dev["url_model"]["batch_size"] < full["url_model"]["batch_size"]
    assert dev["url_model"]["max_train_rows"] is not None
    assert full["url_model"]["max_train_rows"] is None


def test_both_profiles_list_the_same_fusion_baselines():
    dev = load_config(CONFIG_DIR / "dev.yaml")
    full = load_config(CONFIG_DIR / "full.yaml")
    required = {"concatenation_mlp", "probability_average", "weighted_average"}
    assert required <= set(dev["fusion"]["baselines"])
    assert required <= set(full["fusion"]["baselines"])


def test_calibration_is_fitted_on_validation():
    for name in ("dev.yaml", "full.yaml"):
        cfg = load_config(CONFIG_DIR / name)
        assert cfg["fusion"]["calibration"]["fit_on"] == "val"
        assert "temperature" in cfg["fusion"]["calibration"]["compare"]


def test_resolve_paths_handles_relative_config_path(tmp_path, monkeypatch):
    """Regression: a relative --config made every path relative to configs/."""
    cfg = load_config(CONFIG_DIR / "dev.yaml")
    monkeypatch.chdir(BACKEND_ROOT)

    # Simulate exactly what the CLI does: a relative config path.
    paths = resolve_paths(cfg, Path("configs/dev.yaml"))

    assert paths["splits_dir"] == (BACKEND_ROOT / "data" / "splits").resolve()
    assert paths["reports_dir"] == (BACKEND_ROOT / "reports").resolve()
    assert paths["checkpoints_dir"] == (BACKEND_ROOT / "checkpoints").resolve()
    for p in paths.values():
        assert p.is_absolute()


def test_resolve_paths_finds_the_real_dataset():
    cfg = load_config(CONFIG_DIR / "dev.yaml")
    paths = resolve_paths(cfg, CONFIG_DIR / "dev.yaml")
    assert paths["dataset_csv"].is_file(), (
        f"configured dataset_csv does not exist: {paths['dataset_csv']}"
    )
    assert paths["splits_dir"].is_dir(), (
        f"configured splits_dir does not exist: {paths['splits_dir']}"
    )


def test_collection_settings_are_conservative():
    for name in ("dev.yaml", "full.yaml"):
        cfg = load_config(CONFIG_DIR / name)
        col = cfg["collection"]
        assert col["max_redirects"] <= 5
        assert col["max_per_domain"] <= 1
        assert col["respect_robots"] is True
        assert col["delay_seconds"] >= 1.0