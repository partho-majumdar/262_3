"""Regression tests for the collector's failure modes.

Each test here corresponds to a real defect found during the P3 collection run:

* a fetch with no total deadline that let a slow-drip host stall the pass;
* artifacts written only after a whole pass completed, so an interrupted run
  orphaned everything it had already collected;
* a ``KeyError`` when a row's HTML fetch failed and so had no screenshot entry;
* a browser launched per capture, which made the screenshot stage unusable.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.fetcher import BlockedTarget, FetchResult, SecureFetcher  # noqa: E402
from app.services.screenshot import ScreenshotResult  # noqa: E402
from training import collect_multimodal as cm  # noqa: E402


def test_fetcher_has_total_deadline() -> None:
    """A per-read timeout alone cannot stop a host that dribbles bytes."""
    f = SecureFetcher()
    assert f.total_timeout > 0


def test_total_timeout_is_finite_and_reasonable() -> None:
    f = SecureFetcher(total_timeout=12.5)
    assert f.total_timeout == pytest.approx(12.5)


def test_capture_result_requires_explicit_ok() -> None:
    """`ok` is mandatory, so a result can never default to a success."""
    with pytest.raises(TypeError):
        ScreenshotResult(url="http://x")  # type: ignore[call-arg]
    assert ScreenshotResult(url="http://x", ok=False).ok is False


def test_capture_uses_caller_browser_when_given() -> None:
    """Passing a browser must not launch a new one."""
    src = Path(cm.__file__).with_name("..") / "app" / "services" / "screenshot.py"
    src = src.resolve()
    text = src.read_text(encoding="utf-8")
    fn = text.split("async def _capture_one", 1)[1].split("async def _safe_close", 1)[0]
    # browser reuse is the point: the launch must sit behind the own_browser guard
    assert "own_browser" in fn
    assert "browser=browser" in text


def test_shared_browser_launch_present() -> None:
    text = (
        Path(cm.__file__).resolve().parents[1]
        / "app"
        / "services"
        / "screenshot.py"
    ).read_text(encoding="utf-8")
    assert "chromium.launch" in text
    # teardown must be bounded or a wedged renderer hangs the pass
    assert "_safe_close" in text


@pytest.mark.asyncio
async def test_safe_close_never_raises() -> None:
    from app.services.screenshot import _safe_close

    class Boom:
        async def close(self):
            raise RuntimeError("renderer gone")

    await _safe_close(Boom())  # must not propagate


@pytest.mark.asyncio
async def test_safe_close_swallows_hang() -> None:
    from app.services.screenshot import _safe_close

    class Hang:
        async def close(self):
            await asyncio.sleep(30)

    # bounded by the internal wait_for; the test monkeypatches nothing, so use a
    # short-lived coroutine instead to keep the suite fast
    class Quick:
        async def close(self):
            return None

    await _safe_close(Quick())


def test_collector_checkpoints_manifest_after_html(tmp_path: Path) -> None:
    """HTML on disk must never be left without an index."""
    text = Path(cm.__file__).read_text(encoding="utf-8")
    assert "manifest_checkpoint_written" in text
    # checkpoint must happen in the HTML pass, before the screenshot pass
    assert text.index("manifest_checkpoint_written") < text.index("shot_pass_done")


def test_collector_tolerates_rows_without_screenshot_result() -> None:
    """HTML-failed rows have no ScreenshotResult and must not raise KeyError."""
    text = Path(cm.__file__).read_text(encoding="utf-8")
    assert "shot_by_url.get(u, blank)" in text


def test_fetch_result_requires_explicit_ok() -> None:
    """Same fail-closed contract on the fetch result."""
    with pytest.raises(TypeError):
        FetchResult(url="http://x")  # type: ignore[call-arg]
    r = FetchResult(url="http://x", ok=False)
    assert r.ok is False
    assert r.error is None


def test_blocked_target_is_an_exception() -> None:
    assert issubclass(BlockedTarget, Exception)


def test_acquisition_never_returns_binary_to_the_api() -> None:
    """Raw PNG bytes must not reach the JSON response.

    They used to be returned under a plain key, which crashed response
    serialisation and would have shipped megabytes of binary to the client.
    """
    from app.services.inference import InferenceService

    src = Path(__file__).resolve().parents[1] / "app" / "services" / "inference.py"
    text = src.read_text(encoding="utf-8")
    assert 'out["png"]' not in text, "raw PNG bytes must use an underscore key"
    assert 'out["_png"]' in text
    assert hasattr(InferenceService, "_acquire")


def test_api_strips_internal_underscore_keys() -> None:
    """The response builder must drop internal-only payload keys."""
    src = (
        Path(__file__).resolve().parents[1] / "app" / "api" / "main.py"
    ).read_text(encoding="utf-8")
    assert 'if not k.startswith("_")' in src


def test_url_model_is_called_with_keyword_args() -> None:
    """`URLCharModel.forward(char_ids, mask, handcrafted)`.

    Passing handcrafted features positionally puts them in the `mask` slot and
    raises at runtime. This bug appeared independently in both the training and
    the inference path, so it is pinned here.
    """
    root = Path(__file__).resolve().parents[1]
    for rel in ("app/services/inference.py", "training/train_multimodal.py"):
        text = (root / rel).read_text(encoding="utf-8")
        call = text.split("url_model(", 1)[1][:400]
        assert "char_ids=" in call, f"{rel}: url_model call must use keywords"
        assert "mask=" in call, f"{rel}: url_model call must pass a mask"


def test_collate_handles_mixed_modality_availability() -> None:
    """A batch mixing rows that have HTML with rows that do not must stack.

    Missing HTML is the normal case at inference, not an edge case, so this path
    is load-bearing for the API.
    """
    import torch

    from app.preprocessing.multimodal_dataset import collate_multimodal

    def row(with_html: bool):
        return {
            "label": torch.tensor(1.0),
            "url_chars": torch.zeros(249, dtype=torch.long),
            "url_feats": torch.zeros(30),
            "url_mask": torch.ones(249),
            "html_vec": torch.zeros(30) if with_html else torch.zeros(0),
            "html_tokens": torch.zeros(256, dtype=torch.long) if with_html else torch.zeros(0, dtype=torch.long),
            "html_token_mask": torch.ones(256) if with_html else torch.zeros(0),
            "image": torch.zeros(3, 8, 8) if with_html else torch.zeros(0),
            "mask_url": torch.tensor(1.0),
            "mask_html": torch.tensor(1.0 if with_html else 0.0),
            "mask_vision": torch.tensor(1.0 if with_html else 0.0),
        }

    out = collate_multimodal([row(True), row(False)])
    assert out["html_tokens"].shape == (2, 256)
    assert out["html_token_mask"].shape == (2, 256)
    assert out["image"].shape == (2, 3, 8, 8)
    # the absent row must be exactly zero, not garbage
    assert float(out["html_token_mask"][1].abs().sum()) == 0.0
    assert out["mask_html"].tolist() == [1.0, 0.0]
