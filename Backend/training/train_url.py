"""P2 - train the URL CharCNN-BiLSTM model and write real metrics.

Contract:
  * reads the real split from ``data/splits/`` (never re-splits internally)
  * deterministic given ``--seed``
  * checkpoint save/resume, early stopping on validation phishing recall
  * class-weighted loss (no row duplication)
  * temperature calibration fitted on validation, reported against Platt and
    isotonic
  * every metric written to ``reports/`` as JSON, produced by this run only

Usage
-----
    python training/train_url.py --config configs/dev.yaml
    python training/train_url.py --config configs/full.yaml --seed 7
    python training/train_url.py --config configs/dev.yaml --leakage-experiment
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from app.core.config import resolve_device
from app.core.logging_config import configure_logging, get_logger
from app.models.url_model import URLCharModel
from app.preprocessing.url_dataset import (
    SplitData,
    URLDataset,
    load_splits,
    make_loader,
    subsample,
)
from app.preprocessing.url_features import FEATURE_NAMES, FeatureScaler, extract_batch
from app.preprocessing.url_preprocessing import CharTokenizer, suggest_max_length
from app.utils.metrics import (
    calibration_report,
    compute_binary_metrics,
    bootstrap_metric_ci,
)

log = get_logger("train_url")


# ---------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def resolve_paths(cfg: dict, config_dir: Path) -> dict[str, Path]:
    """Resolve config paths against the **backend root**.

    Config paths are documented as relative to ``backend/``. The backend root is
    taken from this module's own location rather than from the config path:
    deriving it from ``config_dir.parent`` is wrong twice over - the config file's
    parent is ``configs/``, and a relative ``--config`` has no absolute parent at
    all until it is resolved.
    """
    root = BACKEND_ROOT
    paths = cfg["paths"]
    out = {}
    for key, default in (
        ("dataset_csv", root / ".." / "Dataset" / "Phishing_URL_Dataset.csv"),
        ("splits_dir", root / "data" / "splits"),
        ("checkpoints_dir", root / "checkpoints"),
        ("reports_dir", root / "reports"),
    ):
        raw = paths.get(key)
        p = Path(raw) if raw else default
        if not p.is_absolute():
            p = (root / p).resolve()
        out[key] = p
    return out


def class_weights(labels: np.ndarray) -> torch.Tensor:
    """Inverse-frequency weights, normalised to mean 1.

    Used with ``BCEWithLogitsLoss(pos_weight=...)``-style weighting so the
    minority class contributes comparable gradient without duplicating rows.
    """
    n_pos = float(np.sum(labels == 1))
    n_neg = float(np.sum(labels == 0))
    if n_pos == 0 or n_neg == 0:
        raise ValueError("training split must contain both classes")
    w_pos = n_neg / n_pos
    return torch.tensor([1.0, w_pos], dtype=torch.float32)


class WeightedBCE(nn.Module):
    """Binary cross-entropy on the logit with a per-class weight."""

    def __init__(self, pos_weight: float) -> None:
        super().__init__()
        self.register_buffer("weight", torch.tensor([1.0, float(pos_weight)]))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.view(-1)
        per_sample = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        w = self.weight[targets.long()]
        return (per_sample * w).mean()


# ---------------------------------------------------------------------------
@dataclass
class TrainState:
    best_val_metric: float = -1.0
    best_epoch: int = -1
    epochs_without_improvement: int = 0
    history: list[dict[str, Any]] = field(default_factory=list)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
) -> dict[str, float]:
    model.train()
    total_loss, n = 0.0, 0
    for char_ids, mask, hand, labels in loader:
        char_ids = char_ids.to(device)
        mask = mask.to(device)
        labels = labels.to(device)
        hand = hand.to(device) if hand.shape[1] else None

        optimizer.zero_grad()
        out = model(char_ids, mask, hand)
        loss = criterion(out.logit, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += float(loss.item()) * labels.size(0)
        n += labels.size(0)
    return {"loss": total_loss / max(n, 1)}


@torch.no_grad()
def predict(
    model: nn.Module,
    loader: DataLoader,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(logits, labels)`` for a split, in loader order."""
    model.eval()
    logits: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for char_ids, mask, hand, y in loader:
        out = model(char_ids.to(device), mask.to(device), hand.to(device) if hand.shape[1] else None)
        logits.append(out.logit.detach().cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(logits), np.concatenate(labels)


# ---------------------------------------------------------------------------
def run_training(
    cfg: dict,
    config_dir: Path,
    seed: int = 42,
    resume: bool = True,
    max_epochs_override: int | None = None,
    tag: str = "url",
    adversarial_augment: float = 0.0,
) -> dict[str, Any]:
    """Train, evaluate and persist. Returns the metrics dict (also written)."""
    set_seed(seed)
    paths = resolve_paths(cfg, config_dir)
    device = resolve_device(cfg.get("runtime", {}).get("device"))
    um = cfg["url_model"]

    log.info(
        "training_start", tag=tag, device=device, seed=seed,
        splits_dir=str(paths["splits_dir"]), config=str(config_dir),
    )

    splits = load_splits(paths["splits_dir"])
    max_rows = um.get("max_train_rows")
    train_split = subsample(splits["train"], max_rows, seed=seed)
    val_split = splits["val"]
    test_split = splits["test"]

    # Adversarial augmentation touches the TRAIN split only. Perturbing val or
    # test would leak the attack families into evaluation and make the robustness
    # numbers meaningless, so those are left exactly as loaded.
    adv_families = None
    if adversarial_augment > 0:
        from training.adversarial_eval import FAMILIES, augment_for_training

        # The normaliser already neutralises these families, so augmented copies of
        # them are near-duplicates of the clean examples. Weighting them would only
        # dilute the semantic attacks that actually break the model.
        adv_families = [f for f in FAMILIES if not f.reversible_by_normaliser]
        n_before = len(train_split)
        train_split = augment_for_training(
            train_split,
            families=adv_families,
            ratio=adversarial_augment,
            seed=seed,
        )
        log.info(
            "adversarial_augmentation",
            train_before=n_before,
            train_after=len(train_split),
            added=len(train_split) - n_before,
            ratio=adversarial_augment,
            families=[f.name for f in adv_families],
        )

    log.info(
        "split_sizes",
        train=len(train_split), val=len(val_split), test=len(test_split),
        train_phishing=int(train_split.labels.sum()),
        val_phishing=int(val_split.labels.sum()),
        test_phishing=int(test_split.labels.sum()),
    )

    # --- tokenizer fitted on TRAINING URLs ONLY ---------------------------
    tokenizer = CharTokenizer(
        max_length=um.get("max_length", 256),
        min_count=5,
    )
    observed_len = suggest_max_length(train_split.urls)
    tokenizer.max_length = min(tokenizer.max_length, max(64, observed_len))
    tokenizer.fit(train_split.urls)
    log.info("tokenizer_ready", vocab_size=tokenizer.vocab_size,
             max_length=tokenizer.max_length, suggested_from_p99=observed_len)

    # --- handcrafted scaler fitted on TRAINING URLs ONLY ------------------
    use_hand = bool(um.get("handcrafted_branch", True))
    scaler = None
    if use_hand:
        rows = extract_batch(train_split.urls)
        scaler = FeatureScaler().fit(rows)

    train_ds = URLDataset(train_split, tokenizer, scaler)
    val_ds = URLDataset(val_split, tokenizer, scaler)
    test_ds = URLDataset(test_split, tokenizer, scaler)

    bs = int(um.get("batch_size", 32))
    nw = int(cfg.get("runtime", {}).get("num_workers", 0))
    train_loader = make_loader(train_ds, bs, shuffle=True, num_workers=nw, seed=seed)
    # Batch 1 for eval keeps memory flat and preserves exact ordering.
    val_loader = make_loader(val_ds, 512, shuffle=False, num_workers=0)
    test_loader = make_loader(test_ds, 512, shuffle=False, num_workers=0)

    model = URLCharModel(
        vocab_size=tokenizer.vocab_size,
        max_length=tokenizer.max_length,
        embedding_dim=int(um.get("embedding_dim", 64)),
        cnn_channels=int(um.get("cnn_channels", 48)),
        cnn_kernel_sizes=tuple(um.get("cnn_kernel_sizes", [3, 5, 7])),
        lstm_hidden=int(um.get("lstm_hidden", 48)),
        lstm_layers=int(um.get("lstm_layers", 1)),
        bidirectional=bool(um.get("bidirectional", True)),
        embedding_out=int(um.get("embedding_out", 128)),
        dropout=float(um.get("dropout", 0.35)),
        n_handcrafted=len(FEATURE_NAMES),
        use_handcrafted=use_hand,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    weights = class_weights(train_split.labels)
    criterion = WeightedBCE(pos_weight=float(weights[1]))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(um.get("lr", 1e-3)),
        weight_decay=float(um.get("weight_decay", 1e-4)),
    )

    ckpt_dir = paths["checkpoints_dir"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"{tag}_model.pt"
    state = TrainState()
    start_epoch = 0

    if resume and ckpt_path.is_file():
        blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(blob["model_state"])
        if "optimizer_state" in blob:
            optimizer.load_state_dict(blob["optimizer_state"])
        if "train_state" in blob:
            saved = blob["train_state"]
            state.best_val_metric = float(saved.get("best_val_metric", -1.0))
            state.best_epoch = int(saved.get("best_epoch", -1))
            state.history = list(saved.get("history", []))
            start_epoch = int(saved.get("next_epoch", 0))
        tokenizer = CharTokenizer.from_state_dict(blob["tokenizer"])
        if blob.get("scaler"):
            scaler = FeatureScaler.from_state_dict(blob["scaler"])
        log.info("resumed_from_checkpoint", path=str(ckpt_path), start_epoch=start_epoch,
                 best_val_metric=state.best_val_metric)

    epochs = int(max_epochs_override or um.get("epochs", 5))
    patience = int(um.get("early_stopping_patience", 2))
    best_test_metrics: dict[str, Any] | None = None
    t0 = time.perf_counter()

    for epoch in range(start_epoch, epochs):
        ep_start = time.perf_counter()
        tr = train_epoch(model, train_loader, criterion, optimizer, device)

        val_logits, val_labels = predict(model, val_loader, device)
        val_probs = 1.0 / (1.0 + np.exp(-val_logits))
        val_m = compute_binary_metrics(
            val_labels, val_probs,
            recall_at_fpr_points=cfg.get("evaluation", {}).get("recall_at_fpr_points", (0.01, 0.05, 0.1)),
        )
        # Select on phishing recall: this is the metric the system exists to
        # maximise, and it is computed on validation only.
        improved = val_m.recall > state.best_val_metric
        record = {
            "epoch": epoch,
            "train_loss": round(tr["loss"], 6),
            "val_recall": round(val_m.recall, 6),
            "val_precision": round(val_m.precision, 6),
            "val_specificity": round(val_m.specificity, 6),
            "val_f1": round(val_m.f1, 6),
            "val_roc_auc": None if np.isnan(val_m.roc_auc) else round(val_m.roc_auc, 6),
            "val_pr_auc": None if np.isnan(val_m.pr_auc) else round(val_m.pr_auc, 6),
            "val_brier": round(val_m.brier, 6),
            "seconds": round(time.perf_counter() - ep_start, 2),
            "is_best": improved,
        }
        state.history.append(record)
        log.info("epoch_complete", **{k: v for k, v in record.items() if k != "epoch"}, epoch=epoch)

        if improved:
            state.best_val_metric = val_m.recall
            state.best_epoch = epoch
            state.epochs_without_improvement = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "tokenizer": tokenizer.state_dict(),
                    "scaler": scaler.state_dict() if scaler else None,
                    "model_config": model.config(),
                    "train_state": {
                        "best_val_metric": state.best_val_metric,
                        "best_epoch": state.best_epoch,
                        "next_epoch": epoch + 1,
                        "history": state.history,
                    },
                    "feature_names": list(FEATURE_NAMES) if use_hand else [],
                    "config": cfg,
                    "seed": seed,
                    "tag": tag,
                },
                ckpt_path,
            )
        else:
            state.epochs_without_improvement += 1
            if state.epochs_without_improvement >= patience:
                log.info("early_stopping", epoch=epoch, best_epoch=state.best_epoch,
                         best_val_recall=state.best_val_metric)
                break

    # --- reload the best checkpoint before evaluating ---------------------
    if ckpt_path.is_file():
        blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(blob["model_state"])
        # Recompute validation logits from the reloaded weights. The loop above
        # left `val_logits` holding the *last* epoch's scores, so reusing them
        # would fit the calibrator on a different model than the one the test
        # metrics and the shipped checkpoint come from.
        val_logits, val_labels = predict(model, val_loader, device)

    # --- test evaluation --------------------------------------------------
    test_logits, test_labels = predict(model, test_loader, device)
    cal = calibration_report(val_labels, val_logits, test_labels, test_logits)

    from app.utils.metrics import TemperatureScaler

    sel = cal["selected_on_validation_brier"]
    if sel == "temperature":
        from app.utils.metrics import TemperatureScaler as _TS

        calibrator = _TS().from_state_dict(cal["methods"]["temperature"]["parameter"])
    elif sel == "platt":
        a = cal["methods"]["platt"]["parameter"]["a"]
        b = cal["methods"]["platt"]["parameter"]["b"]
        calibrator = ("platt", a, b)
    else:
        calibrator = ("uncalibrated",)

    raw_probs = 1.0 / (1.0 + np.exp(-test_logits))
    if sel == "temperature":
        cal_probs = calibrator.transform_logits(test_logits)
    elif sel == "platt":
        _, a, b = calibrator
        cal_probs = 1.0 / (1.0 + np.exp(-(a * test_logits + b)))
    else:
        cal_probs = raw_probs

    final = compute_binary_metrics(
        test_labels, cal_probs,
        recall_at_fpr_points=cfg.get("evaluation", {}).get("recall_at_fpr_points", (0.01, 0.05, 0.1)),
    )
    uncal = compute_binary_metrics(test_labels, raw_probs, recall_at_fpr_points=(0.01, 0.05, 0.1))

    n_boot = int(cfg.get("evaluation", {}).get("bootstrap_samples", 200))
    conf = float(cfg.get("evaluation", {}).get("bootstrap_confidence", 0.95))
    ci = {
        "recall": bootstrap_metric_ci(test_labels, cal_probs, "recall", n_resamples=n_boot, confidence=conf),
        "precision": bootstrap_metric_ci(test_labels, cal_probs, "precision", n_resamples=n_boot, confidence=conf),
        "f1": bootstrap_metric_ci(test_labels, cal_probs, "f1", n_resamples=n_boot, confidence=conf),
        "accuracy": bootstrap_metric_ci(test_labels, cal_probs, "accuracy", n_resamples=n_boot, confidence=conf),
        "recall_at_fpr_0.01": bootstrap_metric_ci(
            test_labels, cal_probs, "recall_at_fpr", n_resamples=max(n_boot // 2, 50),
            confidence=conf, recall_at_fpr_point=0.01,
        ),
    }

    total_seconds = time.perf_counter() - t0
    best_phish = float(np.max(cal_probs[test_labels == 1])) if np.any(test_labels == 1) else float("nan")

    metrics: dict[str, Any] = {
        "run": {
            "tag": tag,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "seed": seed,
            "device": device,
            "torch_version": torch.__version__,
            "config_file": str(config_dir),
            "profile": cfg.get("profile"),
            "train_seconds": round(total_seconds, 2),
            "n_parameters": int(n_params),
        },
        "data": {
            "splits_dir": str(paths["splits_dir"]),
            "train_rows": len(train_split),
            "val_rows": len(val_split),
            "test_rows": len(test_split),
            "train_phishing": int(train_split.labels.sum()),
            "val_phishing": int(val_split.labels.sum()),
            "test_phishing": int(test_split.labels.sum()),
            "max_train_rows_cap": max_rows,
            "subsampled": len(train_split) < len(splits["train"]),
        },
        "tokenizer": {
            "vocab_size": tokenizer.vocab_size,
            "max_length": tokenizer.max_length,
            "suggested_from_p99": observed_len,
            "alphabet_size": len(tokenizer.alphabet),
        },
        "features": {
            "handcrafted_branch": use_hand,
            "n_handcrafted": len(FEATURE_NAMES) if use_hand else 0,
            "feature_names": list(FEATURE_NAMES) if use_hand else [],
        },
        "training": {
            "epochs_requested": epochs,
            "epochs_completed": len(state.history),
            "best_epoch": state.best_epoch,
            "best_val_phishing_recall": state.best_val_metric,
            "early_stopped": len(state.history) < epochs,
            "pos_weight": float(weights[1]),
            "class_weighting": um.get("class_weighting"),
            # Recorded in the written report too, not just the returned dict, so a
            # metrics file is self-describing: without it, url_model.pt and
            # url_adv_model.pt are indistinguishable from their scores alone.
            "adversarial_augment_ratio": adversarial_augment,
            "n_train_urls": int(len(train_split)),
            "history": state.history,
        },
        "test_metrics_calibrated": final.to_dict(),
        "test_metrics_uncalibrated": uncal.to_dict(),
        "calibration": cal,
        "bootstrap_ci_95": ci,
        "checkpoint": str(ckpt_path),
        "confidence_definition": "max(p_phishing, 1 - p_phishing) on the calibrated probability",
    }

    reports_dir = paths["reports_dir"]
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_path = reports_dir / f"metrics_{tag}.json"
    out_path.write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")

    log.info(
        "evaluation_complete",
        tag=tag,
        test_recall=round(final.recall, 6),
        test_precision=round(final.precision, 6),
        test_f1=round(final.f1, 6),
        test_roc_auc=None if np.isnan(final.roc_auc) else round(final.roc_auc, 6),
        test_pr_auc=None if np.isnan(final.pr_auc) else round(final.pr_auc, 6),
        test_brier=round(final.brier, 6),
        calibration_selected=sel,
        adversarial_augment_ratio=adversarial_augment,
        n_train_urls=int(len(train_split)),
        checkpoint=str(ckpt_path),
        metrics_json=str(out_path),
    )
    return metrics


# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Train the URL CharCNN-BiLSTM model.")
    p.add_argument("--config", type=Path, default=BACKEND_ROOT / "configs" / "dev.yaml")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None, help="override the config epoch count")
    p.add_argument("--tag", default="url", help="checkpoint/report name suffix")
    p.add_argument("--no-resume", action="store_true", help="ignore any existing checkpoint")
    p.add_argument("--leakage-experiment", action="store_true",
                   help="train on PhiUSIIL's page-derived engineered columns (experiment only)")
    p.add_argument("--adversarial-augment", type=float, default=0.0,
                   help="fraction of TRAIN URLs to duplicate under a semantic evasion "
                        "family (0 disables; val/test are never perturbed)")
    args = p.parse_args(argv)

    configure_logging(fmt="json")
    cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else int(cfg.get("runtime", {}).get("seed", 42))

    if args.leakage_experiment:
        from leakage_experiment import run_leakage_experiment

        return run_leakage_experiment(cfg, args.config, seed=seed, epochs=args.epochs)

    if not 0.0 <= args.adversarial_augment <= 1.0:
        p.error("--adversarial-augment must be a fraction in [0, 1]")

    # Augmenting changes the training distribution, so a resumed checkpoint from a
    # clean run is not a valid starting point for an augmented one.
    resume = not args.no_resume and args.adversarial_augment == 0.0

    metrics = run_training(
        cfg, args.config, seed=seed, resume=resume,
        max_epochs_override=args.epochs, tag=args.tag,
        adversarial_augment=args.adversarial_augment,
    )
    tm = metrics["test_metrics_calibrated"]
    print(json.dumps({
        "test_recall": tm["recall"],
        "test_precision": tm["precision"],
        "test_f1": tm["f1"],
        "test_specificity": tm["specificity"],
        "test_roc_auc": tm["roc_auc"],
        "test_pr_auc": tm["pr_auc"],
        "test_mcc": tm["mcc"],
        "test_brier": tm["brier"],
        "recall_at_fpr": tm["recall_at_fpr"],
        "metrics_json": str(BACKEND_ROOT / "reports" / f"metrics_{args.tag}.json"),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())