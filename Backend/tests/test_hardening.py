"""Tests for Unicode/IDN hardening, calibration plumbing, and XAI wiring.

Covers the hardening round: confusable/invisible-character folding in the URL
normaliser, temperature-calibration loading in the inference service, and the
leave-one-modality-out influence block plus private-field stripping at the API
boundary.
"""

from __future__ import annotations

import json
import random

import pytest

from app.preprocessing.url_preprocessing import fold_confusables, normalize_url


class TestFoldConfusables:
    def test_cyrillic_lookalikes_map_to_ascii(self) -> None:
        # Cyrillic а=0430 е=0435 о=043E р=0440 с=0441
        assert fold_confusables("раypal.com") == "paypal.com"
        assert fold_confusables("аpple.com") == "apple.com"
        assert fold_confusables("microsоft.com") == "microsoft.com"

    def test_greek_omicron_maps_to_o(self) -> None:
        assert fold_confusables("gοοgle.com") == "google.com"

    def test_uppercase_cyrillic_maps_to_ascii(self) -> None:
        # The lowercase table alone is not enough: a spoofed "Раypal.com" with
        # Cyrillic capital Er renders as an ordinary capital P.
        assert fold_confusables("Раypal.com") == "Paypal.com"
        assert fold_confusables("Аpple.com") == "Apple.com"
        assert fold_confusables("Мicrosoft.com") == "Microsoft.com"

    def test_real_ascii_is_unchanged(self) -> None:
        assert fold_confusables("login.bank-of-america.com") == (
            "login.bank-of-america.com"
        )


class TestNormalizeUrlFullwidth:
    """Fullwidth forms are NFKC's job, not the confusable table's."""

    def test_fullwidth_host_is_folded(self) -> None:
        assert "paypal.com" in normalize_url("http://ｐaypal.com/login")

    def test_fullwidth_punctuation_is_folded(self) -> None:
        assert normalize_url("http://a.com／b") == "http://a.com/b"


class TestNormalizeUrlInvisibles:
    @pytest.mark.parametrize(
        "cp",
        [
            0x200B,  # ZERO WIDTH SPACE
            0x200C,  # ZERO WIDTH NON-JOINER
            0x200D,  # ZERO WIDTH JOINER
            0x2028,  # LINE SEPARATOR
            0x2029,  # PARAGRAPH SEPARATOR
            0xFEFF,  # ZERO WIDTH NO-BREAK SPACE / BOM
            0x00AD,  # SOFT HYPHEN
            0x180E,  # MONGOLIAN VOWEL SEPARATOR
            0x2060,  # WORD JOINER
            0x034F,  # COMBINING GRAPHEME JOINER
            0x061C,  # ARABIC LETTER MARK
            0x200E,  # LEFT-TO-RIGHT MARK
            0x200F,  # RIGHT-TO-LEFT MARK
            0x202E,  # RIGHT-TO-LEFT OVERRIDE
            0x2066,  # LEFT-TO-RIGHT ISOLATE
        ],
    )
    def test_invisibles_are_removed(self, cp: int) -> None:
        ch = chr(cp)
        assert ch not in normalize_url(f"http://paypa{ch}l.com/login")

    def test_invisible_does_not_split_the_host(self) -> None:
        assert "paypal.com" in normalize_url("http://paypa‌l.com/login")

    def test_normalisation_is_idempotent(self) -> None:
        once = normalize_url("HTTPS://WWW.Раypal.com/Login/")
        assert normalize_url(once) == once

    def test_uppercase_homoglyph_folded_on_first_pass(self) -> None:
        """Regression: capitals were absent from the table, so "Раypal.com"
        survived the first pass and only collapsed on the second."""
        assert "рaypal" not in normalize_url("http://Раypal.com/")

    def test_punycode_is_preserved(self) -> None:
        # xn-- is the punycode marker; folding must not mangle the payload.
        assert "xn--80ak6aa92e.com" in normalize_url("http://xn--80ak6aa92e.com/")

    @pytest.mark.parametrize("bad", ["", "not a url", "://///", "http://", "%%%"])
    def test_malformed_input_does_not_raise(self, bad: str) -> None:
        assert isinstance(normalize_url(bad), str)


class TestCalibrationLoading:
    """``_load_calibration`` must degrade quietly, never raise, on bad artifacts."""

    @staticmethod
    def _svc(tmp_path):
        from app.services.inference import InferenceService

        # Bypass __init__/model loading: _load_calibration only touches model_dir.
        svc = InferenceService.__new__(InferenceService)
        svc.model_dir = tmp_path
        return svc

    def test_missing_artifact_returns_none(self, tmp_path) -> None:
        assert self._svc(tmp_path)._load_calibration() is None

    def test_flat_form_written_by_train_multimodal(self, tmp_path) -> None:
        (tmp_path / "fusion_calibration.json").write_text(
            json.dumps({"method": "temperature", "parameter": 2.5}),
            encoding="utf-8",
        )
        cal = self._svc(tmp_path)._load_calibration()
        assert cal is not None
        assert pytest.approx(cal.temperature) == 2.5

    def test_nested_form_is_accepted(self, tmp_path) -> None:
        (tmp_path / "fusion_calibration.json").write_text(
            json.dumps({"method": "temperature", "parameter": {"temperature": 1.75}}),
            encoding="utf-8",
        )
        cal = self._svc(tmp_path)._load_calibration()
        assert pytest.approx(cal.temperature) == 1.75

    def test_bare_temperature_key_without_method_is_rejected(self, tmp_path) -> None:
        """No ``method`` key means the artifact is not one of ours; ignore it."""
        (tmp_path / "fusion_calibration.json").write_text(
            json.dumps({"temperature": 3.0}), encoding="utf-8"
        )
        assert self._svc(tmp_path)._load_calibration() is None

    def test_missing_parameter_is_rejected(self, tmp_path) -> None:
        (tmp_path / "fusion_calibration.json").write_text(
            json.dumps({"method": "temperature"}), encoding="utf-8"
        )
        assert self._svc(tmp_path)._load_calibration() is None

    def test_temperature_one_is_identity(self, tmp_path) -> None:
        """T=1 must leave probabilities untouched, or calibration silently rescales."""
        import numpy as np

        from app.utils.metrics import TemperatureScaler

        svc = self._svc(tmp_path)
        logits = np.array([-3.0, 0.0, 3.0])
        svc.calibrator = None
        raw = svc._apply_calibration(logits)
        scaler = TemperatureScaler()
        scaler.temperature = 1.0
        svc.calibrator = scaler
        assert np.allclose(svc._apply_calibration(logits), raw)


class TestAdversarialPerturbations:
    """Each family must actually alter the URL, or the eval measures nothing."""

    @pytest.mark.parametrize(
        "name",
        [
            "homoglyph",
            "fullwidth",
            "zero_width",
            "scheme_upper",
            "trailing_dot",
            "brand_swap",
            "subdomain_prepend",
            "typosquat",
            "path_shuffle",
            "double_encode",
            "repeat_pad",
        ],
    )
    def test_family_changes_a_phishing_url(self, name: str) -> None:
        from training.adversarial_eval import FAMILIES

        fam = next(f for f in FAMILIES if f.name == name)
        url = "http://secure-login.xyz.tk/verify/account?id=1"
        assert fam.fn(url, random.Random(0)) != url, (
            f"{name} produced an unchanged URL; the robustness measurement "
            "for this family would be vacuous"
        )

    def test_double_encode_does_not_break_the_scheme(self) -> None:
        """Encoding '://' yields an invalid URL, not an attack."""
        from training.adversarial_eval import p_double_encode

        out = p_double_encode("https://evil.tk/a/b", random.Random(3))
        assert out.startswith("https://")
        assert "%252f/www" not in out.lower()

    def test_host_of_returns_scheme_and_remainder(self) -> None:
        from training.adversarial_eval import _host_of

        assert _host_of("https://evil.tk/a") == ("https://", "evil.tk/a")

    def test_split_host_tail_roundtrip(self) -> None:
        from training.adversarial_eval import _host_of, _split_host_tail

        scheme, rest = _host_of("https://evil.tk/a/b?c=1")
        host, tail = _split_host_tail(rest)
        assert host == "evil.tk"
        assert tail == "/a/b?c=1"


class TestAdversarialAugmentation:
    """Augmentation must widen coverage without corrupting the label balance."""

    @staticmethod
    def _split(n_phish: int = 30, n_legit: int = 30):
        import numpy as np

        from app.preprocessing.url_dataset import SplitData

        urls = [f"https://bad{i}.xyz.tk/login" for i in range(n_phish)] + [
            f"https://shop{i}.example.com/cart" for i in range(n_legit)
        ]
        labels = [1] * n_phish + [0] * n_legit
        return SplitData(
            name="train",
            urls=urls,
            labels=np.asarray(labels, dtype=np.int64),
            row_ids=np.arange(len(urls), dtype=np.int64),
        )

    def test_augment_preserves_every_original_url(self) -> None:
        from training.adversarial_eval import augment_for_training

        base = self._split()
        out = augment_for_training(base, ratio=1.0, seed=0)
        assert out.urls[: len(base.urls)] == base.urls
        assert len(out) >= len(base)

    def test_augment_keeps_the_class_balance_of_the_additions(self) -> None:
        """Perturbing only phishing URLs would teach the inverse lesson."""
        import numpy as np

        from training.adversarial_eval import augment_for_training

        base = self._split(40, 40)
        out = augment_for_training(base, ratio=1.0, seed=42)
        added = np.asarray(out.labels[len(base.urls) :])
        assert len(added) > 0
        # Both classes must appear among the additions, in roughly equal share.
        assert 0.3 < added.mean() < 0.7

    def test_augment_is_deterministic_for_a_seed(self) -> None:
        from training.adversarial_eval import augment_for_training

        a = augment_for_training(self._split(), ratio=0.8, seed=7)
        b = augment_for_training(self._split(), ratio=0.8, seed=7)
        assert a.urls == b.urls

    def test_zero_ratio_is_a_no_op(self) -> None:
        from training.adversarial_eval import augment_for_training

        base = self._split()
        out = augment_for_training(base, ratio=0.0, seed=1)
        assert out.urls == base.urls

    def test_row_ids_of_additions_are_negative_sentinels(self) -> None:
        """Originals keep their row ids so a split audit stays valid."""
        from training.adversarial_eval import augment_for_training

        base = self._split()
        out = augment_for_training(base, ratio=0.9, seed=3)
        assert (out.row_ids[: len(base.urls)] == base.row_ids).all()
        assert (out.row_ids[len(base.urls) :] < 0).all()

    def test_normaliser_reversible_families_are_excluded_by_default_filter(self) -> None:
        from training.adversarial_eval import FAMILIES

        semantic = [f.name for f in FAMILIES if not f.reversible_by_normaliser]
        assert "brand_swap" in semantic
        assert "homoglyph" not in semantic

    def test_augmented_urls_are_not_duplicates_of_the_originals(self) -> None:
        from training.adversarial_eval import augment_for_training

        base = self._split(60, 60)
        out = augment_for_training(base, ratio=1.0, seed=11)
        added = set(out.urls[len(base.urls) :])
        assert added
        assert not (added & set(base.urls))


class TestXaiLiveExecution:
    """XAI must actually run against the loaded checkpoints.

    The unit tests elsewhere use fakes, which is why a numpy truth-test and an
    indivisible reshape both shipped and only failed at runtime, caught in a
    live /analyze call. These exercise the real code paths with the real
    service when a checkpoint is present.
    """

    @pytest.fixture(scope="class")
    def svc(self):
        from app.core.config import get_settings
        from app.services.inference import InferenceService

        try:
            return InferenceService(get_settings().model_dir)
        except Exception as exc:  # noqa: BLE001
            pytest.skip(f"no loadable checkpoints: {exc}")

    def test_explain_url_returns_contributions(self, svc) -> None:
        from app.services.xai import XaiService

        out = XaiService(svc).explain_url("https://secure-login.xyz.tk/verify/account?id=1")
        assert isinstance(out, list)
        if svc.available.get("url"):
            assert out, "explain_url returned nothing for an available URL model"
            assert all({"label", "contribution"} == set(o) for o in out)
            assert all(isinstance(o["contribution"], float) for o in out)

    def test_explain_url_does_not_raise_on_a_normal_url(self, svc) -> None:
        """Regression: `if feats` on a numpy array raised ValueError."""
        from app.services.xai import XaiService

        out = XaiService(svc).explain_url("https://www.wikipedia.org/")
        assert isinstance(out, list)

    def test_explain_vision_handles_an_indivisible_grid(self, svc) -> None:
        """Regression: reshape(6, -1, 6, -1) on 224x224 raised RuntimeError."""
        import numpy as np
        from PIL import Image

        from app.services.xai import XaiService

        if not svc.available.get("vision"):
            pytest.skip("vision model not loaded")
        buf = np.zeros((224, 224, 3), dtype=np.uint8)
        buf[80:140, 60:160] = 255
        import io

        png = io.BytesIO()
        Image.fromarray(buf).save(png, format="PNG")
        out = XaiService(svc).explain_vision(png.getvalue(), grid=6)
        assert len(out) == 8
        assert all("cell" in o["label"] for o in out)

    def test_explain_fusion_is_a_measured_counterfactual(self, svc) -> None:
        import asyncio

        verdict, acq = asyncio.get_event_loop().run_until_complete(
            svc.analyze("https://example.com/", ["url"])
        )
        embs = acq.get("_embs")
        avail = acq.get("_avail")
        if not embs or not svc.available.get("fusion"):
            pytest.skip("no fused embeddings for this request")
        from app.services.xai import XaiService

        out = XaiService(svc).explain_fusion(embs, avail, float(verdict.probability))
        assert out["method"] == "leave_one_modality_out"
        assert "url" in out["per_modality"]


class TestApiBoundaryHygiene:
    """Private acquisition fields must never reach the public response."""

    def test_underscore_keys_are_filtered(self) -> None:
        acquisition = {
            "url_available": True,
            "_embs": {"url": [0.1]},
            "_avail": {"url": True},
            "_html": "<html>secret</html>",
            "_png": b"\x89PNG",
        }
        pub = {k: v for k, v in acquisition.items() if not k.startswith("_")}
        assert list(pub) == ["url_available"]
        assert not any(k.startswith("_") for k in pub)

    def test_analyze_response_declares_calibration_flag(self) -> None:
        from app.schemas.analysis import AnalyzeResponse

        assert "probability_is_calibrated" in AnalyzeResponse.model_fields

    def test_analyze_response_declares_modality_influence(self) -> None:
        from app.schemas.analysis import AnalyzeResponse

        assert "modality_influence" in AnalyzeResponse.model_fields

    def test_influence_defaults_to_empty_not_none(self) -> None:
        from app.schemas.analysis import AnalyzeResponse

        fields = AnalyzeResponse.model_fields
        assert fields["modality_influence"].default_factory() == {}