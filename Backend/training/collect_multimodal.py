"""P3 collector: acquire page HTML and rendered screenshots for a URL subset.

Produces ``data/manifest.csv`` — the join key that lets P4/P5/P6 train on
exactly the rows whose artifacts exist — plus stored artifacts under
``data/html/`` and ``data/screenshots/``.

A note on missingness, which is the main methodological risk in this phase
-----------------------------------------------------------------------------
PhiUSIIL was crawled around 2022. Most phishing hosts are long dead, so the
*fetch success rate will differ strongly by class*. If phishing pages fail to
load far more often than legitimate ones, then "did the page load" is itself a
label predictor, and a fusion model can score well by reading the mask instead
of the content.

That is not a bug to hide: a deployed detector genuinely does know whether a
page loaded. But it must be **measured and reported**, because it changes what
the multimodal numbers mean. The collector therefore always prints the
success rate per class and the correlation between availability and label, and
those figures go into the P3 report.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.core.logging_config import configure_logging, get_logger  # noqa: E402
from app.preprocessing.url_dataset import load_splits  # noqa: E402
from app.services.fetcher import FetchResult, SecureFetcher, store_html  # noqa: E402
from app.services.screenshot import (  # noqa: E402
    SCREENSHOT_HARDENING_ARGS,
    ScreenshotResult,
    capture_screenshot,
)
from training.train_url import load_config, resolve_paths  # noqa: E402

log = get_logger(__name__)


def select_subset(labels: np.ndarray, urls: list[str], n: int, seed: int) -> np.ndarray:
    """Class-stratified indices, so both classes survive small ``n``."""
    if n <= 0 or n >= len(labels):
        return np.arange(len(labels))
    rng = np.random.default_rng(seed)
    keep: list[int] = []
    for cls in (0, 1):
        idx = np.where(labels == cls)[0]
        if idx.size == 0:
            continue
        quota = min(int(round(n * idx.size / labels.size)), idx.size)
        keep.extend(rng.choice(idx, size=quota, replace=False).tolist())
    return np.array(sorted(keep), dtype=np.int64)


async def collect(
    rows: pd.DataFrame,
    cfg: dict,
    paths: dict[str, Path],
    *,
    do_html: bool,
    do_screenshot: bool,
    limit: int | None = None,
) -> pd.DataFrame:
    c = cfg.get("collection", {})
    fetcher = SecureFetcher(
        timeout=float(c.get("request_timeout", 10)),
        connect_timeout=float(c.get("connect_timeout", 5)),
        max_response_size_mb=float(c.get("max_response_size_mb", 5)),
        max_redirects=int(c.get("max_redirects", 5)),
        user_agent=str(c.get("user_agent", "MMRESearchBot/1.0")),
        delay_seconds=float(c.get("delay_seconds", 1.0)),
        max_per_domain=int(c.get("max_per_domain", 1)),
    )
    data_root = paths["splits_dir"].parent
    html_dir = data_root / "html"
    shot_dir = data_root / "screenshots"
    urls = rows["url"].tolist()
    if limit:
        urls = urls[:limit]
        rows = rows.iloc[:limit]
    log.info("collect_start", n=len(urls), html=do_html, screenshot=do_screenshot)

    # --- HTML pass --------------------------------------------------------
    if do_html:
        sem = asyncio.Semaphore(int(c.get("concurrency", 8)))
        stored: dict[str, tuple[bool, str, str, int]] = {}
        done = 0

        async def one(u: str):
            nonlocal done
            async with sem:
                try:
                    # Hard outer bound: whatever the cause, a fetch that has not
                    # finished in this long is recorded as a failed acquisition
                    # and abandoned. Without it a single wedged socket stalls the
                    # entire pass.
                    res = await asyncio.wait_for(fetcher.fetch(u), timeout=75)
                except asyncio.TimeoutError:
                    res = FetchResult(url=u, ok=False, error="collector_deadline_exceeded")
                except Exception as exc:  # noqa: BLE001 - a bad URL is a mask
                    res = FetchResult(url=u, ok=False, error=f"{type(exc).__name__}: {exc}"[:200])
            # Store as each result lands rather than after the whole pass, so a
            # long run makes visible progress instead of appearing stuck.
            path = store_html(res, html_dir) if res.ok else None
            stored[u] = (
                bool(res.ok),
                str(path) if path else "",
                res.error or "",
                res.status_code or 0,
            )
            done += 1
            if done % 100 == 0:
                log.info(
                    "html_progress",
                    done=done,
                    ok=sum(1 for v in stored.values() if v[0]),
                )
            return res

        await asyncio.gather(*(one(u) for u in urls))

        rows = rows.assign(
            html_ok=[stored[u][0] for u in urls],
            html_path=[stored[u][1] for u in urls],
            html_error=[stored[u][2] for u in urls],
            http_status=[stored[u][3] for u in urls],
        )
        ok_html = [stored[u][0] for u in urls]
        log.info(
            "html_pass_done",
            ok=int(sum(ok_html)), total=len(ok_html),
            rate=round(float(np.mean(ok_html)), 4) if ok_html else 0.0,
        )
        # Checkpoint immediately: the HTML pass is the slow, failure-prone half,
        # and without this an interrupted screenshot pass would orphan every
        # collected page on disk with no index describing them.
        checkpoint = rows.assign(
            shot_ok=False, shot_path="", shot_error="not_attempted", shot_bytes=0
        )
        checkpoint.to_csv(data_root / "manifest.csv", index=False)
        log.info("manifest_checkpoint_written", path=str(data_root / "manifest.csv"))

    # --- screenshot pass -------------------------------------------------
    if do_screenshot:
        # Only render hosts that answered once. Re-navigating a URL already
        # known to be dead wastes a browser launch per row and buys nothing,
        # since a host that refuses HTTP will not paint either.
        if "html_ok" in rows.columns:
            targets = rows.loc[rows["html_ok"], "url"].tolist()
            log.info("shot_targets", n=len(targets), skipped=int((~rows["html_ok"]).sum()))
        else:
            targets = urls
        if not targets:
            rows = rows.assign(shot_ok=False, shot_path="", shot_error="html_fetch_failed",
                               shot_bytes=0)
            return rows

        sem = asyncio.Semaphore(int(c.get("screenshot_concurrency", 3)))
        c = cfg.get("collection", {})

        shots_done = 0
        # One browser for the whole pass: launching Chromium per URL cost
        # several seconds each and made the screenshot stage the bottleneck.
        from playwright.async_api import async_playwright

        async with async_playwright() as pw:
            browser = await pw.chromium.launch(
                headless=True, args=SCREENSHOT_HARDENING_ARGS,
                chromium_sandbox=False, timeout=20_000,
            )

            async def shot(u: str):
                nonlocal shots_done
                async with sem:
                    res = await capture_screenshot(
                        u,
                        shot_dir,
                        width=1280,
                        height=800,
                        nav_timeout_ms=15_000,
                        user_agent=str(cfg.get("collection", {}).get("user_agent", "MMRESearchBot/1.0")),
                        browser=browser,
                    )
                shots_done += 1
                if shots_done % 50 == 0:
                    log.info("screenshot_progress", done=shots_done, total=len(targets))
                return res

            try:
                shots = list(await asyncio.gather(*(shot(u) for u in targets)))
            finally:
                try:
                    await asyncio.wait_for(browser.close(), timeout=15)
                except Exception:  # noqa: BLE001
                    pass
        shot_by_url = {u: s for u, s in zip(targets, shots)}
        # Rows whose HTML fetch failed were never sent to the browser, so they
        # have no ScreenshotResult at all - they must be filled in explicitly
        # rather than looked up blindly.
        blank = ScreenshotResult(url="", ok=False, error="html_fetch_failed")

        def shot_of(u: str):
            return shot_by_url.get(u, blank)

        rows = rows.assign(
            shot_ok=[shot_of(u).ok for u in urls],
            shot_path=[str(shot_of(u).png_path) if shot_of(u).png_path else "" for u in urls],
            shot_error=[shot_of(u).error or "" for u in urls],
            shot_bytes=[shot_of(u).png_bytes for u in urls],
        )
        log.info(
            "shot_pass_done",
            attempted=len(targets),
            ok=int(sum(s.ok for s in shots)),
            rate=round(float(np.mean([s.ok for s in shots])), 4) if shots else 0.0,
        )
    return rows


def missingness_report(rows: pd.DataFrame) -> dict:
    """Availability broken down by class — the confound this phase can create."""
    out: dict = {"n": int(len(rows))}
    for col in ("html_ok", "shot_ok"):
        if col not in rows.columns:
            continue
        out[col] = {
            "overall_rate": float(rows[col].mean()) if len(rows) else 0.0,
            "by_label": {
                str(int(lbl)): float(g[col].mean())
                for lbl, g in rows.groupby("label")
            },
        }
        a = rows.loc[rows["label"] == 1, col].astype(float)
        b = rows.loc[rows["label"] == 0, col].astype(float)
        if len(a) and len(b):
            out[col]["availability_gap_phishing_minus_legit"] = float(a.mean() - b.mean())
    if "html_ok" in rows and "shot_ok" in rows:
        out["n_with_both_modalities"] = int((rows["html_ok"] & rows["shot_ok"]).sum())
    return out


def render_report(rep: dict, rows: pd.DataFrame, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "# P3 - Multimodal acquisition report",
        "",
        f"Generated {datetime.now(timezone.utc).isoformat()} by "
        "`training/collect_multimodal.py`.",
        "",
        "## What was collected",
        "",
        f"- URLs attempted: **{rep['n']}**",
    ]
    for col, label in (("html_ok", "HTML"), ("shot_ok", "Screenshot")):
        if col not in rep:
            continue
        d = rep[col]
        by = d["by_label"]
        lines += [
            f"- **{label}** fetched: {d['overall_rate']:.1%} overall "
            f"({by.get('1', 0):.1%} phishing / {by.get('0', 0):.1%} legitimate); "
            f"availability gap {d['availability_gap_phishing_minus_legit']:+.1%}",
        ]
    if "n_with_both_modalities" in rep:
        lines.append(f"- Rows with **both** modalities: {rep['n_with_both_modalities']}")

    lines += [
        "",
        "## The missingness confound",
        "",
        "PhiUSIIL was crawled in 2022, so phishing hosts have largely expired.",
        "If phishing pages fail to load more often than legitimate ones, then",
        "*availability alone* predicts the label, and a fusion model can score",
        "well by reading the modality mask instead of the content.",
        "",
        "This is measured above rather than assumed. A deployed detector really",
        "does know whether a page loaded, so the mask is a legitimate input - but",
        "it means the multimodal headline numbers partly reward link rot, and they",
        "are **not** comparable to a deployment where every URL resolves. Any",
        "comparison against the URL-only model must be read with this in mind.",
        "",
        "## Isolation posture",
        "",
        "Docker is **not installed on this host**, so the containerised screenshot",
        "service described in the deployment contract was not built or verified.",
        "Screenshots were captured in-process with Chromium hardened from the",
        "inside: a fresh isolated browser context per page, all non-http(s)",
        "schemes aborted at the request layer (blocking `file://` reads and",
        "`ftp://`/`ws://` exfiltration), all permissions denied, JS dialogs",
        "auto-dismissed, downloads refused, and a hard navigation timeout.",
        "",
        "**What this does not provide:** no filesystem, PID or network namespace",
        "around the browser. A Chromium vulnerability would run with the",
        "collector's privileges. This is a genuine reduction in security posture",
        "and the containerised service remains the only form suitable for",
        "untrusted traffic.",
        "",
        "## Artifacts",
        "",
        "Stored under `backend/data/` which is git-ignored; no fetched HTML or",
        "screenshot is committed.",
        "",
    ]
    path = out_dir / "collection_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Collect HTML + screenshots for a URL subset.")
    p.add_argument("--config", type=Path, default=BACKEND_ROOT / "configs" / "dev.yaml")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n", type=int, default=200, help="stratified subset size (0 = all)")
    p.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    p.add_argument("--html", action="store_true", default=True)
    p.add_argument("--no-html", dest="html", action="store_false")
    p.add_argument("--screenshot", action="store_true", default=True)
    p.add_argument("--no-screenshot", dest="screenshot", action="store_false")
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args(argv)

    configure_logging(fmt="json")
    cfg = load_config(args.config)
    paths = resolve_paths(cfg, args.config)
    splits = load_splits(paths["splits_dir"])

    frames = []
    for name in args.splits:
        sp = splits[name]
        sub = select_subset(sp.labels, sp.urls, args.n, args.seed)
        frames.append(
            pd.DataFrame(
                {
                    "row_id": sp.row_ids[sub],
                    "split": name,
                    "url": [sp.urls[i] for i in sub],
                    "label": sp.labels[sub],
                }
            )
        )
    rows = pd.concat(frames, ignore_index=True)
    log.info("subset_selected", n=int(len(rows)), splits=args.splits)

    rows = asyncio.run(
        collect(
            rows, cfg, paths,
            do_html=args.html, do_screenshot=args.screenshot, limit=args.limit,
        )
    )

    manifest = paths["splits_dir"].parent / "manifest.csv"
    rows.to_csv(manifest, index=False)
    rep = missingness_report(rows)
    report_path = render_report(rep, rows, paths["reports_dir"])
    (paths["reports_dir"] / "collection_report.json").write_text(
        json.dumps(rep, indent=2), encoding="utf-8"
    )

    print(json.dumps({**rep, "manifest": str(manifest), "report": str(report_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
