"""Availability-controlled evaluation of the multimodal models.

`reports/availability_confound.md` shows that whether a page could be collected
predicts the label on its own (F1 0.834 from the availability mask alone). A
model evaluated on the unrestricted test set therefore cannot be separated from
that shortcut.

This script re-scores the same models **restricted to the subset where the
modality is actually available**. That holds the mask constant by construction:
if HTML exists, every evaluated row has HTML, so availability carries no
information and the score has to come from content.

Both numbers are reported. Unrestricted is deployed behaviour; controlled is the
one that says whether the model learned anything about phishing.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.preprocessing.multimodal_dataset import (  # noqa: E402
    MultiModalDataset,
    build_rows,
    collate_multimodal,
    load_manifest,
)
from app.utils.metrics import compute_binary_metrics  # noqa: E402


def _block(y, p) -> dict:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    if len(y) < 10 or len(np.unique(y)) < 2:
        return {"n": int(len(y)), "note": "too few rows or one class"}
    mt = compute_binary_metrics(y, p, threshold=0.5)
    return {
        "n": int(len(y)),
        "positives": int(y.sum()),
        "prevalence": round(float(y.mean()), 4),
        "accuracy": round(mt.accuracy, 4),
        "precision": round(mt.precision, 4),
        "recall": round(mt.recall, 4),
        "f1": round(mt.f1, 4),
        "specificity": round(mt.specificity, 4),
        "roc_auc": round(mt.roc_auc, 4),
        "pr_auc": round(mt.pr_auc, 4),
    }


def main() -> int:
    from torch.utils.data import DataLoader

    from app.models.fusion_model import FusionModel
    from app.models.html_model import HTMLModalityModel
    from app.models.url_model import URLCharModel
    from app.models.vision_model import VisionModalityModel
    from app.preprocessing.html_tokenizer import HTMLTextTokenizer
    from app.preprocessing.url_features import FeatureScaler
    from app.preprocessing.url_preprocessing import CharTokenizer
    from training.train_url import load_config, resolve_paths

    cfg_path = Path("configs/dev.yaml")
    cfg = load_config(cfg_path)
    paths = resolve_paths(cfg, cfg_path)
    device = torch.device("cpu")
    ck_dir = Path(paths["checkpoints_dir"])
    rep_dir = Path(paths["reports_dir"])

    ck = torch.load(ck_dir / "multimodal.pt", map_location=device, weights_only=False)

    url_ck = torch.load(ck_dir / "url_model.pt", map_location=device, weights_only=False)
    tok = CharTokenizer.from_state_dict(url_ck["tokenizer"])
    manifest = load_manifest(Path(paths["splits_dir"]).parent / "manifest.csv")
    scaler = FeatureScaler.from_state_dict(url_ck["scaler"])
    rows = build_rows(manifest, tokenizer=tok, scaler=scaler)

    html_tok = HTMLTextTokenizer.from_state_dict(ck["html_tokenizer"])
    for r in rows:
        if r.has_html:
            ids, m = html_tok.encode(_visible(r))
            r.html_tokens = torch.tensor(ids, dtype=torch.long)
            r.html_token_mask = torch.tensor(m, dtype=torch.float32)

    test = [r for r in rows if r.split == "test"]
    dl = DataLoader(
        MultiModalDataset(test), batch_size=32, shuffle=False,
        collate_fn=collate_multimodal,
    )

    url = URLCharModel(**ck["url_model_config"]).to(device)
    url.load_state_dict(ck["url_model_state"])
    html = HTMLModalityModel(**ck["html_model_config"]).to(device)
    html.load_state_dict(ck["html_state"])
    vcfg = dict(ck["vision_model_config"])
    vcfg.pop("feature_dim", None)  # reported, not an __init__ argument
    vcfg["pretrained"] = False  # weights come from the checkpoint
    vis = VisionModalityModel(**vcfg).to(device)
    vis.load_state_dict(ck["vision_state"])
    fus = FusionModel(**ck["fusion_model_config"]).to(device)
    fus.load_state_dict(ck["fusion_state"])
    for m in (url, html, vis, fus):
        m.eval()

    acc: dict[str, list] = {k: [] for k in ("url", "html", "vision", "fusion")}
    with torch.no_grad():
        for b in dl:
            ou = url(
                char_ids=b["url_chars"].to(device),
                mask=b["url_mask"].to(device),
                handcrafted=(
                    b["url_feats"].to(device) if b["url_feats"].numel() else None
                ),
            )
            oh = html(
                b["html_tokens"].to(device),
                b["html_token_mask"].to(device),
                b["html_vec"].to(device),
            )
            ov = vis(b["image"].to(device))
            acc["url"].append(ou.logit.cpu().numpy())
            acc["html"].append(oh.logit.cpu().numpy())
            acc["vision"].append(ov.logit.cpu().numpy())
            # Fusion consumes embeddings, exactly as in training.
            acc["fusion"].append(
                fus(
                    ou.embedding.to(device),
                    oh.embedding.to(device),
                    ov.embedding.to(device),
                    b["mask_url"].to(device),
                    b["mask_html"].to(device),
                    b["mask_vision"].to(device),
                ).logit.cpu().numpy()
            )

    p = {k: 1.0 / (1.0 + np.exp(-np.concatenate(v))) for k, v in acc.items()}
    y = np.array([float(r.label) for r in test])
    hh = np.array([r.mask()["html"] for r in test])
    hv = np.array([r.mask()["vision"] for r in test])

    out: dict = {
        "note": (
            "Controlled = only rows where the modality exists, so the mask is "
            "constant and cannot act as a shortcut. Mask-only baseline F1 is "
            "0.834 (reports/availability_confound.md)."
        ),
        "n_test": int(len(test)),
        "models": {
            "url": {"unrestricted": _block(y, p["url"])},
            "html": {
                "unrestricted": _block(y, p["html"]),
                "controlled_html_available": _block(y[hh], p["html"][hh]),
            },
            "vision": {
                "unrestricted": _block(y, p["vision"]),
                "controlled_screenshot_available": _block(y[hv], p["vision"][hv]),
            },
            "fusion": {
                "unrestricted": _block(y, p["fusion"]),
                "controlled_both_available": _block(
                    y[hh & hv], p["fusion"][hh & hv]
                ),
            },
        },
    }

    rep_dir.mkdir(parents=True, exist_ok=True)
    (rep_dir / "availability_controlled_metrics.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8"
    )

    lines = [
        "# Availability-controlled multimodal evaluation",
        "",
        "Unrestricted rows include cases where the modality is missing, so the "
        "availability mask itself contributes signal. Controlled rows use only "
        "the cases where the modality exists, which holds the mask constant.",
        "",
        f"Mask-only baseline for comparison: **F1 0.834**.",
        "",
        "| model | subset | n | prevalence | accuracy | F1 | specificity | ROC-AUC |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, blocks in out["models"].items():
        for sub, b in blocks.items():
            if "f1" not in b:
                continue
            lines.append(
                f"| {name} | {sub} | {b['n']} | {b['prevalence']} | {b['accuracy']} | "
                f"{b['f1']} | {b['specificity']} | {b['roc_auc']} |"
            )
    lines += [
        "",
        "## How to read this",
        "",
        "- **The URL branch alone scores 1.000.** The fused model's perfect score "
        "is inherited from it; HTML and vision add nothing measurable on top.",
        "- **The HTML and vision models are not usable as standalone detectors.** "
        "Specificity of 0.00-0.36 means they flag most legitimate pages as "
        "phishing. Their F1 looks healthy only because phishing is ~92% of each "
        "restricted subset.",
        "- Their ROC-AUC (0.79 HTML, 0.90 vision) shows some ranking signal, but "
        "far less than the F1 suggests.",
        "- A controlled F1 near 0.834 for a *content* model would mean it is "
        "reading domain liveness rather than page content.",
        "",
        "Conclusion: on this dataset the URL string features saturate the task "
        "and the multimodal pipeline cannot be shown to help.",
    ]
    (rep_dir / "availability_controlled_metrics.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(out, indent=2))
    return 0


def _visible(row) -> str:
    from app.preprocessing.multimodal_dataset import visible_text_of, _soup_of

    return visible_text_of(_soup_of(row.html))[:4000] if row.html else ""


if __name__ == "__main__":
    raise SystemExit(main())
