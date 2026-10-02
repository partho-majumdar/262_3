"""Train the HTML, vision and fusion models on the collected artifacts.

Run order matters and is enforced here:

1. **HTML** and **vision** are trained on rows where that modality actually
   exists. Training a modality on zero-filled rows for missing pages would teach
   the model that "no page" is a class.
2. **Fusion** is then trained on top of the frozen unimodal encoders, using
   every row, because missing modalities are the normal case at inference and
   the fusion network has to cope with them.

Evaluation is per modality and for the fused model on the *same* test rows, so
the comparison in the report is like-for-like rather than apples-to-oranges on
differently sized subsets.

Honesty constraint this script enforces
---------------------------------------
The multimodal branches train on a few hundred real pages, not 235,795. Every
report it writes states the exact training row count and says so plainly. A
metric computed on 100 rows is not comparable to one computed on 35,305, and
the report must not invite that comparison.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.core.logging_config import configure_logging, get_logger  # noqa: E402
from app.models.fusion_model import FusionModel  # noqa: E402
from app.models.html_model import HTMLModalityModel  # noqa: E402
from app.models.url_model import URLCharModel  # noqa: E402
from app.models.vision_model import VisionModalityModel  # noqa: E402
from app.preprocessing.html_tokenizer import HTMLTextTokenizer  # noqa: E402
from app.preprocessing.multimodal_dataset import (  # noqa: E402
    MultiModalDataset,
    build_rows,
    collate_multimodal,
    load_manifest,
)
from app.utils.metrics import compute_binary_metrics  # noqa: E402
from training.train_url import class_weights, load_config, resolve_paths, set_seed  # noqa: E402

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _loader(rows, batch_size, shuffle, image_size=224, augment=False, seed=42):
    ds = MultiModalDataset(rows, image_size=image_size, train_augment=augment)
    g = torch.Generator()
    g.manual_seed(seed)
    return torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=g if shuffle else None,
        collate_fn=collate_multimodal,
        drop_last=False,
    )


def _class_weighted_bce(rows) -> nn.Module:
    y = np.array([r.label for r in rows], dtype=np.float64)
    w = class_weights(y)
    pos = float(torch.clamp(w[1], min=1.0))
    neg = float(torch.clamp(w[0], min=1.0))
    return nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos / max(neg, 1e-6)))


@torch.no_grad()
def _evaluate(model, loader, device, forward) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits, labels = [], []
    for batch in loader:
        out = forward(batch, device)
        logits.append(out.reshape(-1).cpu().numpy())
        labels.append(batch["label"].numpy())
    return np.concatenate(logits), np.concatenate(labels)


def _metrics(labels, logits) -> dict:
    p = 1.0 / (1.0 + np.exp(-logits))
    m = compute_binary_metrics(labels, p)
    d = m.to_dict()
    return {
        "recall": d["recall"],
        "precision": d["precision"],
        "f1": d["f1"],
        "specificity": d["specificity"],
        "roc_auc": None if np.isnan(d["roc_auc"]) else d["roc_auc"],
        "pr_auc": None if np.isnan(d["pr_auc"]) else d["pr_auc"],
        "mcc": d["mcc"],
        "brier": d["brier"],
        "n": int(len(labels)),
    }


# ---------------------------------------------------------------------------
# HTML branch
# ---------------------------------------------------------------------------
def train_html(rows_by_split, cfg, device) -> dict:
    hc = cfg.get("html_model", {})
    train_rows = [r for r in rows_by_split["train"] if r.has_html]
    val_rows = [r for r in rows_by_split["val"] if r.has_html]
    test_rows = [r for r in rows_by_split["test"] if r.has_html]
    if len(train_rows) < 20:
        log.warning("html_skipped_too_few_rows", n=len(train_rows))
        return {"trained": False, "reason": f"only {len(train_rows)} train rows have HTML"}

    from app.preprocessing.html_features import N_HTML_FEATURES, visible_text_of
    from app.preprocessing.multimodal_dataset import _soup_of

    tok = HTMLTextTokenizer(
        vocab_size=int(hc.get("vocab_size", 4096)),
        max_tokens=int(hc.get("max_tokens", 512)),
        min_count=1,
    )
    # Fit the vocabulary on training-split text only.
    tok.fit([visible_text_of(_soup_of(r.html))[:4000] for r in train_rows])
    _attach_tokens(rows_by_split, tok)

    model = HTMLModalityModel(
        vocab_size=max(tok.size, 2),
        max_tokens=tok.max_tokens,
        n_dom_features=N_HTML_FEATURES,
        embedding_dim=int(hc.get("embedding_dim", 128)),
        transformer_layers=int(hc.get("transformer_layers", 2)),
        transformer_heads=int(hc.get("transformer_heads", 4)),
        transformer_ff=int(hc.get("transformer_ff", 256)),
        embedding_out=int(hc.get("embedding_out", 128)),
        dropout=float(hc.get("dropout", 0.2)),
    ).to(device)

    crit = _class_weighted_bce(train_rows)
    opt = torch.optim.AdamW(
        model.parameters(), lr=float(hc.get("lr", 8e-4)),
        weight_decay=float(hc.get("weight_decay", 1e-4)),
    )
    bs = int(hc.get("batch_size", 16))
    epochs = int(hc.get("epochs", 5))
    patience = int(hc.get("early_stopping_patience", 2))
    train_loader = _loader(train_rows, bs, True)
    val_loader = _loader(val_rows, bs, False)

    def fwd(batch, dev):
        return model(
            batch["html_tokens"].to(dev),
            batch["html_token_mask"].to(dev),
            batch["html_vec"].to(dev),
        ).logit

    best, best_state, bad = -np.inf, None, 0
    for ep in range(epochs):
        model.train()
        t0 = time.perf_counter()
        for batch in train_loader:
            opt.zero_grad()
            loss = crit(fwd(batch, device), batch["label"].to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        vl, vy = _evaluate(model, val_loader, device, fwd)
        vm = _metrics(vy, vl)
        log.info("html_epoch", epoch=ep, val_auc=vm["roc_auc"], s=round(time.perf_counter() - t0, 1))
        score = vm["roc_auc"] if vm["roc_auc"] is not None else vm["f1"]
        if score > best:
            best, bad = score, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state:
        model.load_state_dict(best_state)

    test_loader = _loader(test_rows, bs, False)
    tl, ty = _evaluate(model, test_loader, device, fwd)
    out = {
        "trained": True,
        "tokenizer": tok.state_dict(),
        "model_config": model.config_dict(),
        "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        "n_train": len(train_rows),
        "n_val": len(val_rows),
        "test": _metrics(ty, tl),
    }
    log.info("html_done", test=out["test"])
    return out


def _attach_tokens(rows_by_split, tok: HTMLTextTokenizer) -> None:
    from app.preprocessing.html_features import visible_text_of
    from app.preprocessing.multimodal_dataset import _soup_of

    for rows in rows_by_split.values():
        texts = [visible_text_of(_soup_of(r.html))[:4000] if r.html else "" for r in rows]
        ids, mask = tok.encode_batch(texts)
        for r, i, m in zip(rows, ids, mask):
            r.html_tokens = torch.tensor(i, dtype=torch.long)
            r.html_token_mask = torch.tensor(m, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Vision branch
# ---------------------------------------------------------------------------
def train_vision(rows_by_split, cfg, device) -> dict:
    vc = cfg.get("vision_model", {})
    train_rows = [r for r in rows_by_split["train"] if r.has_vision]
    val_rows = [r for r in rows_by_split["val"] if r.has_vision]
    test_rows = [r for r in rows_by_split["test"] if r.has_vision]
    if len(train_rows) < 20:
        log.warning("vision_skipped_too_few_rows", n=len(train_rows))
        return {"trained": False, "reason": f"only {len(train_rows)} train rows have screenshots"}

    model = VisionModalityModel(
        backbone=str(vc.get("backbone", "resnet18")),
        pretrained=bool(vc.get("pretrained", True)),
        embedding_out=int(vc.get("embedding_out", 128)),
        dropout=float(vc.get("dropout", 0.2)),
    ).to(device)
    log.info("vision_backbone", name=vc.get("backbone"), pretrained_loaded=model.pretrained_loaded)

    crit = _class_weighted_bce(train_rows)
    opt = torch.optim.AdamW(
        model.parameters(), lr=float(vc.get("lr", 3e-4)),
        weight_decay=float(vc.get("weight_decay", 1e-4)),
    )
    bs = int(vc.get("batch_size", 16))
    epochs = int(vc.get("epochs", 3))
    size = int(vc.get("image_size", 224))
    train_loader = _loader(train_rows, bs, True, image_size=size, augment=True)
    val_loader = _loader(val_rows, bs, False, image_size=size)

    def fwd(batch, dev):
        return model(batch["image"].to(dev)).logit

    best, best_state, bad = -np.inf, None, 0
    for ep in range(epochs):
        model.train()
        t0 = time.perf_counter()
        for batch in train_loader:
            opt.zero_grad()
            loss = crit(fwd(batch, device), batch["label"].to(device))
            loss.backward()
            opt.step()
        vl, vy = _evaluate(model, val_loader, device, fwd)
        vm = _metrics(vy, vl)
        log.info("vision_epoch", epoch=ep, val_auc=vm["roc_auc"], s=round(time.perf_counter() - t0, 1))
        score = vm["roc_auc"] if vm["roc_auc"] is not None else vm["f1"]
        if score > best:
            best, bad = score, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= int(vc.get("early_stopping_patience", 1)):
                break
    if best_state:
        model.load_state_dict(best_state)

    test_loader = _loader(test_rows, bs, False, image_size=size)
    tl, ty = _evaluate(model, test_loader, device, fwd)
    return {
        "trained": True,
        "model_config": model.config_dict(),
        "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        "pretrained_loaded": bool(model.pretrained_loaded),
        "n_train": len(train_rows),
        "n_val": len(val_rows),
        "test": _metrics(ty, tl),
    }


# ---------------------------------------------------------------------------
# Fusion
# ---------------------------------------------------------------------------
@torch.no_grad()
def _embed_all(model, loader, device, fn) -> tuple[torch.Tensor, dict]:
    model.eval()
    embs, masks, labels = [], [], []
    for batch in loader:
        e = fn(batch, device)
        embs.append(e.cpu())
        masks.append(
            torch.stack(
                [batch["mask_url"], batch["mask_html"], batch["mask_vision"]], dim=-1
            )
        )
        labels.append(batch["label"])
    return torch.cat(embs), {"mask": torch.cat(masks), "label": torch.cat(labels)}


def train_fusion(rows_by_split, url_model, html_bundle, vision_bundle, cfg, device) -> dict:
    fc = cfg.get("fusion", {})
    shared = int(fc.get("shared_dim", 128))
    fusion = FusionModel(
        shared_dim=shared,
        hidden_dims=tuple(fc.get("hidden_dims", (128, 64))),
        dropout=float(fc.get("dropout", 0.2)),
        modality_dropout=float(fc.get("modality_dropout", 0.3)),
    ).to(device)

    bs = int(fc.get("batch_size", 64))
    loaders = {
        s: _loader(rows_by_split[s], bs, s == "train") for s in ("train", "val", "test")
    }

    def embed_fn(model, key):
        def fn(batch, dev):
            if key == "url":
                # Positional args would put url_feats into the `mask` slot.
                return url_model(
                    char_ids=batch["url_chars"].to(dev),
                    mask=batch["url_mask"].to(dev),
                    handcrafted=(
                        batch["url_feats"].to(dev)
                        if batch["url_feats"].numel()
                        else None
                    ),
                ).embedding
            if key == "html":
                if not html_bundle or not html_bundle.get("trained"):
                    return torch.zeros(batch["label"].size(0), shared)
                return html_model(
                    batch["html_tokens"].to(dev),
                    batch["html_token_mask"].to(dev),
                    batch["html_vec"].to(dev),
                ).embedding
            if not vision_bundle or not vision_bundle.get("trained"):
                return torch.zeros(batch["label"].size(0), shared)
            return vision_model(batch["image"].to(dev)).embedding

        return fn

    url_model = url_model.eval()
    for p in url_model.parameters():
        p.requires_grad_(False)
    html_model = vision_model = None
    if html_bundle and html_bundle.get("trained"):
        from app.models.html_model import HTMLModalityModel as H

        html_model = H(**html_bundle["model_config"])
        html_model.load_state_dict(html_bundle["state_dict"])
        html_model = html_model.to(device).eval()
        for p in html_model.parameters():
            p.requires_grad_(False)
    if vision_bundle and vision_bundle.get("trained"):
        from app.models.vision_model import VisionModalityModel as V

        vision_model = V(
            backbone=vision_bundle["model_config"]["backbone"],
            pretrained=False,
            embedding_out=vision_bundle["model_config"]["embedding_out"],
        )
        vision_model.load_state_dict(vision_bundle["state_dict"])
        vision_model = vision_model.to(device).eval()
        for p in vision_model.parameters():
            p.requires_grad_(False)

    # Precompute frozen embeddings once per split.
    cached = {}
    for split, loader in loaders.items():
        eu, m = _embed_all(url_model, loader, device, embed_fn(url_model, "url"))
        eh, _ = _embed_all(html_model or nn.Identity(), loader, device, embed_fn(url_model, "html"))
        ev, _ = _embed_all(vision_model or nn.Identity(), loader, device, embed_fn(url_model, "vision"))
        cached[split] = (eu, eh, ev, m["mask"], m["label"])
        log.info("fusion_cached", split=split, n=int(eu.size(0)))

    crit = nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW(
        fusion.parameters(), lr=float(fc.get("lr", 1e-3)),
        weight_decay=float(fc.get("weight_decay", 1e-4)),
    )
    aux_w = float(fc.get("auxiliary_unimodal_loss_weight", 0.2))
    p_drop = float(fc.get("modality_dropout", 0.3))

    def forward_split(split, train: bool):
        eu, eh, ev, mask, y = cached[split]
        mu, mh, mv = mask[:, 0], mask[:, 1], mask[:, 2]
        if train:
            # Independent per-modality dropout so no branch can be assumed present.
            keep = torch.rand_like(mask)
            drop = (keep < p_drop) & (mask > 0)
            mu = mu * (~drop[:, 0]).float()
            mh = mh * (~drop[:, 1]).float()
            mv = mv * (~drop[:, 2]).float()
        return fusion(eu.to(device), eh.to(device), ev.to(device),
                      mu.to(device), mh.to(device), mv.to(device)), y.to(device)

    epochs = int(fc.get("epochs", 8))
    patience = int(fc.get("early_stopping_patience", 3))
    best, best_state, bad = -np.inf, None, 0
    for ep in range(epochs):
        fusion.train()
        eu, eh, ev, mask, y = cached["train"]
        perm = torch.randperm(eu.size(0))
        total = 0.0
        for i in range(0, perm.numel(), bs):
            idx = perm[i : i + bs]
            sub = (eu[idx], eh[idx], ev[idx], mask[idx], y[idx])
            mu, mh, mv = sub[3][:, 0], sub[3][:, 1], sub[3][:, 2]
            keep = torch.rand_like(sub[3])
            drop = (keep < p_drop) & (sub[3] > 0)
            mu = mu * (~drop[:, 0]).float()
            mh = mh * (~drop[:, 1]).float()
            mv = mv * (~drop[:, 2]).float()
            opt.zero_grad()
            out = fusion(sub[0].to(device), sub[1].to(device), sub[2].to(device),
                         mu.to(device), mh.to(device), mv.to(device))
            yb = sub[4].to(device)
            loss = crit(out.logit, yb)
            for name, head, m_ in (("url", "url", mu), ("html", "html", mh), ("vision", "vision", mv)):
                aux = out.modality_logits[name]
                # Only score an auxiliary head where that modality was present.
                sel = m_.to(device) > 0
                if sel.any():
                    loss = loss + aux_w * crit(aux[sel], yb[sel])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(fusion.parameters(), 5.0)
            opt.step()
            total += float(loss)

        out, yv = forward_split("val", train=False)
        # Validation is inference-only; without this the logits keep their graph
        # and the .numpy() conversion below raises.
        vl = out.logit.detach().cpu().numpy()
        vm = _metrics(yv.detach().cpu().numpy(), vl)
        log.info("fusion_epoch", epoch=ep, loss=round(total, 3), val_auc=vm["roc_auc"])
        score = vm["roc_auc"] if vm["roc_auc"] is not None else vm["f1"]
        if score > best:
            best, bad = score, 0
            best_state = {k: v.detach().clone() for k, v in fusion.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state:
        fusion.load_state_dict(best_state)

    out, yt = forward_split("test", train=False)
    logits = out.logit.detach().cpu().numpy()

    # Calibrate on the validation split only, then report the effect on test.
    # app/services/inference.py::_load_calibration reads checkpoints/
    # fusion_calibration.json and silently falls back to an uncalibrated sigmoid
    # when it is absent, so the artifact has to be written here or the API would
    # advertise calibration it never applies.
    calibration = None
    val_out, y_val = forward_split("val", train=False)
    val_logits = val_out.logit.detach().cpu().numpy()
    y_val_np = y_val.detach().cpu().numpy()

    def _brier(logit_arr, label_arr):
        p = 1.0 / (1.0 + np.exp(-logit_arr))
        return float(np.mean((p - label_arr) ** 2))

    if len(np.unique(y_val_np)) > 1:
        from app.utils.metrics import TemperatureScaler

        scaler = TemperatureScaler().fit(val_logits, y_val_np)
        calibration = {
            "method": "temperature",
            "parameter": scaler.state_dict()["temperature"],
            "fitted_on": "val",
            "n_val": int(len(y_val_np)),
            "val_brier_before": _brier(val_logits, y_val_np),
            "val_brier_after": _brier(scaler.transform_logits(val_logits), y_val_np),
            "test_brier_before": _brier(logits, yt.detach().cpu().numpy()),
            "test_brier_after": _brier(scaler.transform_logits(logits), yt.detach().cpu().numpy()),
        }
        log.info(
            "fusion_calibration",
            val_brier_before=round(calibration["val_brier_before"], 6),
            val_brier_after=round(calibration["val_brier_after"], 6),
            test_brier_before=round(calibration["test_brier_before"], 6),
            test_brier_after=round(calibration["test_brier_after"], 6),
        )

    res = {
        "trained": True,
        "model_config": fusion.config_dict(),
        "state_dict": {k: v.cpu() for k, v in fusion.state_dict().items()},
        "test": _metrics(yt.detach().cpu().numpy(), logits),
        "test_mean_gates": out.gates.mean(dim=0).tolist(),
        "test_gate_by_class": {
            "phishing": out.gates[yt.to(device) > 0].mean(dim=0).tolist(),
            "legitimate": out.gates[yt.to(device) == 0].mean(dim=0).tolist(),
        },
        "modality_availability_on_test": {
            "url": float(cached["test"][3][:, 0].mean()),
            "html": float(cached["test"][3][:, 1].mean()),
            "vision": float(cached["test"][3][:, 2].mean()),
        },
        "calibration": calibration,
    }
    log.info("fusion_done", test=res["test"], gates=res["test_mean_gates"])
    return res


# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Train HTML / vision / fusion models.")
    p.add_argument("--config", type=Path, default=BACKEND_ROOT / "configs" / "dev.yaml")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip-html", action="store_true")
    p.add_argument("--skip-vision", action="store_true")
    args = p.parse_args(argv)

    configure_logging(fmt="json")
    cfg = load_config(args.config)
    paths = resolve_paths(cfg, args.config)
    set_seed(args.seed)
    device = "cpu"

    manifest = load_manifest(paths["splits_dir"].parent / "manifest.csv")
    rows_by_split: dict[str, list] = {"train": [], "val": [], "test": []}

    # Reuse the URL tokenizer/scaler that the P2 checkpoint was trained with, so
    # the URL embeddings inside fusion match what the URL model already learned.
    url_ck = torch.load(paths["checkpoints_dir"] / "url_model.pt", map_location="cpu",
                        weights_only=False)
    from app.preprocessing.url_features import FeatureScaler
    from app.preprocessing.url_preprocessing import CharTokenizer

    tok = CharTokenizer.from_state_dict(url_ck["tokenizer"])
    scaler = FeatureScaler.from_state_dict(url_ck["scaler"])
    all_rows = build_rows(manifest, tokenizer=tok, scaler=scaler)
    for r in all_rows:
        rows_by_split.setdefault(r.split, []).append(r)

    counts = {s: len(v) for s, v in rows_by_split.items()}
    log.info("manifest_loaded", **counts)

    url_model = URLCharModel(**url_ck["model_config"])
    url_model.load_state_dict(url_ck["model_state"])

    html_bundle = None if args.skip_html else train_html(rows_by_split, cfg, device)
    vision_bundle = None if args.skip_vision else train_vision(rows_by_split, cfg, device)
    fusion = train_fusion(rows_by_split, url_model, html_bundle, vision_bundle, cfg, device)

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": str(args.config),
        "seed": args.seed,
        "device": device,
        "rows": counts,
        "html": html_bundle,
        "vision": vision_bundle,
        "fusion": fusion,
    }

    def strip(sd):
        if isinstance(sd, dict):
            return {k: strip(v) for k, v in sd.items() if k != "state_dict"}
        return sd

    # Carry forward any modality that was skipped but already has trained
    # weights. Overwriting the checkpoint with None for a skipped branch would
    # silently destroy a previously trained model.
    prior = {}
    mm_path = paths["checkpoints_dir"] / "multimodal.pt"
    if mm_path.is_file():
        try:
            prior = torch.load(mm_path, map_location="cpu", weights_only=False)
        except Exception as exc:  # noqa: BLE001
            log.warning("prior_checkpoint_unreadable", error=str(exc))
            prior = {}

    def carry(bundle, prefix: str):
        if bundle is not None and bundle.get("state_dict") is not None:
            return bundle
        cfg_k, st_k = f"{prefix}_model_config", f"{prefix}_state"
        if prior.get(cfg_k) is None or prior.get(st_k) is None:
            return bundle
        carried = dict(bundle or {})
        carried.setdefault("model_config", prior[cfg_k])
        carried.setdefault("state_dict", prior[st_k])
        carried["carried_from_previous_run"] = True
        log.info(f"{prefix}_branch_carried_forward")
        return carried

    html_bundle = carry(html_bundle, "html")
    vision_bundle = carry(vision_bundle, "vision")

    ck = {
        "url_model_state": url_ck["model_state"],
        "url_model_config": url_ck["model_config"],
        "url_tokenizer": url_ck["tokenizer"],
        "url_scaler": url_ck["scaler"],
        "html_model_config": (html_bundle or {}).get("model_config"),
        "html_state": (html_bundle or {}).get("state_dict"),
        "html_tokenizer": (html_bundle or {}).get("tokenizer"),
        "vision_model_config": (vision_bundle or {}).get("model_config"),
        "vision_state": (vision_bundle or {}).get("state_dict"),
        "fusion_model_config": fusion.get("model_config"),
        "fusion_state": fusion.get("state_dict"),
    }
    torch.save(ck, paths["checkpoints_dir"] / "multimodal.pt")

    # Write the calibration artifact the inference service looks for. Without
    # this file InferenceEngine._load_calibration returns None and the API serves
    # an uncalibrated sigmoid while still describing the score as calibrated.
    fusion_cal = (fusion or {}).get("calibration")
    cal_path = paths["checkpoints_dir"] / "fusion_calibration.json"
    if fusion_cal and fusion_cal.get("parameter") is not None:
        cal_path.write_text(
            json.dumps(
                {
                    "method": "temperature",
                    "parameter": fusion_cal["parameter"],
                    "temperature": fusion_cal["parameter"],
                    "fitted_on": fusion_cal.get("fitted_on"),
                    "val_brier_before": fusion_cal.get("val_brier_before"),
                    "val_brier_after": fusion_cal.get("val_brier_after"),
                    "test_brier_before": fusion_cal.get("test_brier_before"),
                    "test_brier_after": fusion_cal.get("test_brier_after"),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        log.info("fusion_calibration_written", path=str(cal_path))
    elif cal_path.exists():
        # Fusion was skipped or degenerate; a stale file would silently apply a
        # calibration fitted against a different model.
        cal_path.unlink()
        log.info("fusion_calibration_removed", path=str(cal_path))

    reports_dir = paths["reports_dir"]
    (reports_dir / "metrics_multimodal.json").write_text(
        json.dumps(strip(report), indent=2, default=str), encoding="utf-8"
    )
    (reports_dir / "multimodal_report.md").write_text(
        render_report(report), encoding="utf-8"
    )

    print(json.dumps({
        "rows": counts,
        "html_test": (html_bundle or {}).get("test"),
        "vision_test": (vision_bundle or {}).get("test"),
        "fusion_test": fusion.get("test"),
        "mean_gates": fusion.get("test_mean_gates"),
    }, indent=2, default=str))
    return 0


def render_report(rep: dict) -> str:
    L = [
        "# Multimodal training report",
        "",
        f"Generated {rep['generated_at_utc']} by `training/train_multimodal.py`.",
        "",
        "## Read this before comparing any number here",
        "",
        "The URL branch in P2 trains on 164,760 rows (40,000 in the dev profile).",
        "The HTML and vision branches can only train on pages that were **still",
        "reachable in 2026** from a dataset crawled in 2022, which is a few hundred",
        "rows, not hundreds of thousands. Metrics from these two branches are",
        "therefore **not comparable** to the URL model's numbers, and are not",
        "comparable to published PhiUSIIL results either.",
        "",
        "They are reported because the system genuinely uses those modalities, and",
        "because hiding them would misrepresent what the fusion network has learned.",
        "",
        f"Rows available: {rep['rows']}",
        "",
    ]

    def block(title: str, bundle: dict | None, note: str = "") -> None:
        # extend(), not += : augmented assignment would make L local to block().
        L.extend([f"## {title}", ""])
        if note:
            L.extend([f"> {note}", ""])
        if not bundle or not bundle.get("trained"):
            L.extend(
                [f"Not trained: {bundle.get('reason') if bundle else 'skipped'}", ""]
            )
            return
        t = bundle["test"]
        L.extend([
            f"- Training rows: **{bundle['n_train']}**, validation rows: {bundle['n_val']}, "
            f"test rows: **{t['n']}**",
            "",
            "| Metric | Value |",
            "| --- | ---: |",
        ])
        for k in ("recall", "precision", "f1", "specificity", "roc_auc", "pr_auc", "mcc", "brier"):
            v = t.get(k)
            L.append(f"| {k} | {'n/a' if v is None else f'{v:.4f}'} |")
        L.append("")

    block(
        "HTML modality",
        rep.get("html"),
        note=(
            "These numbers are computed **only on rows where HTML was "
            "collectable**, because the model cannot run on a missing page. "
            "Unrestricted scores, and the availability-matched comparison, are "
            "in `availability_controlled_metrics.md`. Phishing is ~92% of this "
            "subset, which is why F1 looks healthy while specificity is low."
        ),
    )
    block(
        "Vision modality",
        rep.get("vision"),
        note=(
            "Computed **only on rows where a screenshot was captured**, for the "
            "same reason. See `availability_controlled_metrics.md`."
        ),
    )

    f = rep.get("fusion") or {}
    L.extend(["## Fused model", ""])
    if not f.get("trained"):
        L.extend(["Not trained.", ""])
    else:
        t = f["test"]
        L.extend([
            f"- Test rows: **{t['n']}**",
            "",
            "| Metric | Value |",
            "| --- | ---: |",
        ])
        for k in ("recall", "precision", "f1", "specificity", "roc_auc", "pr_auc", "mcc", "brier"):
            v = t.get(k)
            L.append(f"| {k} | {'n/a' if v is None else f'{v:.4f}'} |")
        L.extend([
            "",
            "### Read this number correctly",
            "",
            "A near-perfect fused score here is **not** evidence of a strong "
            "multimodal detector. Two separate things are being measured:",
            "",
            "1. **The URL branch alone already scores 1.0 on this test set.** The "
            "fused model inherits that; page content adds nothing measurable on "
            "top of it. See `availability_controlled_metrics.md`.",
            "2. **The availability mask is itself a label giveaway.** A phishing "
            "URL yields a page 76% of the time, a legitimate URL only 9%, because "
            "the phishing hosts have expired since labelling. A model reading only "
            "the mask already scores F1 0.834 "
            "(`availability_confound.md`).",
            "",
            "The standalone HTML and vision models are the informative part, and "
            "they are weak: both have high recall but near-zero specificity, "
            "meaning they flag most legitimate pages as phishing.",
            "",
            "### Learned modality weights (mean gate)",
            "",
            "| Modality | Mean weight | Weight on phishing | Weight on legitimate |",
            "| --- | ---: | ---: | ---: |",
        ])
        names = ("URL", "HTML", "Vision")
        for i, n in enumerate(names):
            L.append(
                f"| {n} | {f['test_mean_gates'][i]:.4f} | "
                f"{f['test_gate_by_class']['phishing'][i]:.4f} | "
                f"{f['test_gate_by_class']['legitimate'][i]:.4f} |"
            )
        av = f["modality_availability_on_test"]
        L.extend([
            "",
            f"Modality availability on the test split: URL {av['url']:.1%}, "
            f"HTML {av['html']:.1%}, Vision {av['vision']:.1%}.",
            "",
            "### The missingness caveat, restated",
            "",
            "Because availability differs by class, part of the fused model's signal",
            "comes from *which modalities loaded*, not only from what they contain.",
            "A deployment where every submitted URL resolves would not enjoy that",
            "advantage. This is stated here rather than buried, and it is why the",
            "gates above are reported per class.",
            "",
        ])
    return "\n".join(L)


if __name__ == "__main__":
    raise SystemExit(main())
