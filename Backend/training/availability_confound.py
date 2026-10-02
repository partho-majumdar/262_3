"""How much of the multimodal signal is just domain liveness?

The PhiUSIIL URLs were labelled around 2018-2020. Most phishing hosts in it are
long dead, while most legitimate sites are still up. That means **modality
availability is correlated with the label**, and a model that reads only the
availability mask can score well without ever looking at a page.

This script measures that baseline so the HTML/vision/fusion numbers can be read
honestly: any modality model that fails to beat `mask_only` has not learned
phishing content, it has learned which URLs still resolve.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.utils.metrics import compute_binary_metrics  # noqa: E402


def mask_only_metrics(m: pd.DataFrame) -> dict:
    """Predict phishing purely from whether HTML was collectable."""
    y = m["label"].to_numpy()
    pred = m["html_ok"].to_numpy().astype(float)
    mt = compute_binary_metrics(y, pred, threshold=0.5)
    return {
        "accuracy": float((pred == y).mean()),
        "precision": mt.precision,
        "recall": mt.recall,
        "f1": mt.f1,
        "specificity": mt.specificity,
    }


def split_availability(m: pd.DataFrame) -> pd.DataFrame:
    g = m.groupby(["split", "label"]).agg(
        n=("label", "size"), html_rate=("html_ok", "mean"), shot_rate=("shot_ok", "mean")
    )
    return g.reset_index()


def main() -> int:
    data_root = Path(__file__).resolve().parents[1] / "data"
    m = pd.read_csv(data_root / "manifest.csv")

    out: dict = {
        "rows": int(len(m)),
        "availability_by_split_label": split_availability(m).to_dict(orient="records"),
        "mask_only_baseline": mask_only_metrics(m),
    }

    # How strong is the correlation, on its own terms?
    y = m["label"].to_numpy()
    h = m["html_ok"].to_numpy().astype(float)
    out["availability_gap_phishing_minus_legit"] = round(
        float(h[y == 1].mean() - h[y == 0].mean()), 4
    )
    out["contingency"] = {
        "html_ok_and_phishing": int(((h == 1) & (y == 1)).sum()),
        "html_ok_and_legit": int(((h == 1) & (y == 0)).sum()),
        "no_html_and_phishing": int(((h == 0) & (y == 1)).sum()),
        "no_html_and_legit": int(((h == 0) & (y == 0)).sum()),
    }

    reports = Path(__file__).resolve().parents[1] / "reports"
    reports.mkdir(exist_ok=True)
    (reports / "availability_confound.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8"
    )

    b = out["mask_only_baseline"]
    lines = [
        "# Availability confound",
        "",
        f"Rows: {out['rows']}",
        "",
        "## Modality availability by class",
        "",
        "| split | label | n | html_ok | shot_ok |",
        "| --- | --- | --- | --- | --- |",
    ]
    for r in out["availability_by_split_label"]:
        lines.append(
            f"| {r['split']} | {int(r['label'])} | {r['n']} | "
            f"{r['html_rate']:.3f} | {r['shot_rate']:.3f} |"
        )
    lines += [
        "",
        f"Availability gap (phishing - legitimate): "
        f"**{out['availability_gap_phishing_minus_legit']:+.3f}**",
        "",
        "## Baseline: predict phishing iff HTML was collectable",
        "",
        "| metric | value |",
        "| --- | --- |",
        f"| accuracy | {b['accuracy']:.4f} |",
        f"| precision | {b['precision']:.4f} |",
        f"| recall | {b['recall']:.4f} |",
        f"| f1 | {b['f1']:.4f} |",
        "",
        "## Reading",
        "",
        "A model that never opens a page and only reads the availability mask "
        f"already reaches F1 {b['f1']:.3f}. Any HTML, vision, or fusion model "
        "must be compared against this number, not against 0.5. A result near it "
        "means the model is reading domain liveness, not phishing content.",
        "",
        "Cause: the dataset labels are from 2018-2020 and most phishing hosts in "
        "it have since expired, while most legitimate sites are still live. This "
        "is a property of the benchmark, not a bug in the collector, and it "
        "cannot be fixed by collecting more pages from the same URLs.",
    ]
    (reports / "availability_confound.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps(out["mask_only_baseline"], indent=2))
    print("gap:", out["availability_gap_phishing_minus_legit"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
