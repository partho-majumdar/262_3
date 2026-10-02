"""P-GNN - train the domain/IP phishing GCN and write real metrics.

Contract:
  * reads the real grouped split from ``data/splits/`` and never re-splits rows
  * node-level train/val/test assignment is derived from that same grouped split,
    so no registered domain can be both a training node and a test node
  * the graph is built by ``app.preprocessing.graph_build``, which forbids label
    leakage into edge construction
  * reports the majority-class baseline next to the model, because a GCN that
    cannot beat "predict the majority class" has told us nothing
  * every metric in ``reports/`` is produced by this run

READ THE LIMITATION IN THE REPORT
----------------------------------
The graph is built from **URL strings only**. There is no DNS resolution, so
almost no domain has an IP edge and the graph is overwhelmingly
domain<->domain structural edges. Treat every number below as a floor for the
graph branch, not as evidence that infrastructure sharing works.

Usage
-----
    python training/train_gnn.py --config configs/dev.yaml
    python training/train_gnn.py --config configs/dev.yaml --max-nodes 4000 --epochs 30
    python training/train_gnn.py --config configs/full.yaml --seed 7
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402

from app.core.logging_config import configure_logging, get_logger  # noqa: E402
from app.models.graph_model import PhishingGCN  # noqa: E402
from app.preprocessing.graph_build import (  # noqa: E402
    EDGE_TYPE_NAMES,
    EDGE_DOMAIN_IP,
    EDGE_DOMAIN_SUBNET,
    GRAPH_FEATURE_NAMES,
    N_EDGE_TYPES,
    SPLIT_TEST,
    SPLIT_TRAIN,
    SPLIT_VAL,
    GraphSpec,
    build_graph,
    default_graph_path,
    save_graph,
)
from app.preprocessing.url_dataset import SplitData, load_splits, subsample  # noqa: E402
from app.utils.metrics import compute_binary_metrics  # noqa: E402
from training.train_url import load_config, resolve_paths, set_seed  # noqa: E402

log = get_logger("train_gnn")

#: Always emitted in the report. Kept as a module constant so the JSON, the
#: markdown and this source file cannot drift apart.
LIMITATION = (
    "The graph is built from URL strings only - there is NO live DNS resolution. "
    "Only URLs whose host is already an IPv4 literal produce an IP node, so the "
    "IP/subnet relations are almost empty and the graph is dominated by "
    "domain<->domain structural edges (shared subdomain token, shared TLD, token "
    "containment). These metrics therefore measure how much a GNN can do with "
    "name structure alone. Adding passive/historical DNS data (or a resolver) is "
    "the single highest-value change to this branch."
)

#: Config block read from ``configs/*.yaml``. Absent today - the configs are not
#: ours to edit - so every value falls back to the default below.
CFG_SECTION = "graph_model"
CFG_DEFAULTS: dict[str, Any] = {
    "hidden_dim": 64,
    "n_layers": 3,
    "dropout": 0.5,
    "head_dropout": 0.3,
    "input_dropout": 0.1,
    "lr": 0.01,
    "weight_decay": 5e-4,
    "epochs": 200,
    "early_stopping_patience": 25,
    "max_nodes": 6000,
    "max_tld_degree": 8,
    "max_shared_degree": 32,
    "grad_clip": 5.0,
}


# ---------------------------------------------------------------------------
class WeightedBCE(nn.Module):
    """Binary cross-entropy on the node logit with a phishing-class weight.

    Same shape as the URL branch's loss: weighting rather than row duplication,
    because duplicating a node would also duplicate it in the graph and change
    the neighbourhood the model sees.
    """

    def __init__(self, pos_weight: float) -> None:
        super().__init__()
        self.register_buffer("weight", torch.tensor([1.0, float(pos_weight)]))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # ``targets`` stays float for the BCE itself and is only narrowed to
        # integer *indices* when gathering the per-class weight.
        targets_f = targets.view(-1)
        per_sample = nn.functional.binary_cross_entropy_with_logits(
            logits, targets_f, reduction="none"
        )
        return (per_sample * self.weight[targets_f.long()]).mean()


@dataclass
class TrainState:
    """Early-stopping bookkeeping (mirrors ``train_url.TrainState``)."""

    best_val_loss: float = float("inf")
    best_epoch: int = -1
    epochs_without_improvement: int = 0
    history: list[dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Sampling / split verification
# ---------------------------------------------------------------------------
def sample_splits(
    splits: dict[str, SplitData], max_nodes: int | None, seed: int
) -> tuple[list[str], np.ndarray, list[str], dict[str, int]]:
    """Concatenate the splits, optionally capping total URLs.

    The cap is applied **per split in proportion to its original size**, and
    within a split via the shared class-stratified ``subsample``. Both choices
    matter: a naive ``urls[:cap]`` would read the tail of ``train.txt`` and could
    silently drop the minority class, and capping only the train split would
    leave val/test out of the graph entirely.

    Returns:
        ``(urls, labels, split_names, sizes)`` with ``sizes`` the per-split URL
        counts actually used.
    """
    order = ("train", "val", "test")
    total = sum(len(splits[s]) for s in order)
    urls: list[str] = []
    labels: list[np.ndarray] = []
    names: list[str] = []
    sizes: dict[str, int] = {}
    for s in order:
        part = splits[s]
        if max_nodes is None or total <= max_nodes:
            capped = part
        else:
            quota = int(round(max_nodes * len(part) / total))
            capped = subsample(part, max(quota, 1), seed=seed)
        sizes[s] = len(capped)
        urls.extend(capped.urls)
        labels.append(np.asarray(capped.labels, dtype=np.int64))
        names.extend([s] * len(capped))
    return urls, np.concatenate(labels) if labels else np.zeros(0, np.int64), names, sizes


def verify_split_grouping(
    spec: GraphSpec, domain_split_csv: Path
) -> dict[str, Any]:
    """Cross-check node splits against ``data/splits/domain_split.csv``.

    The grouping guarantee we need is: *one registered domain occupies exactly
    one split*. ``make_splits.py`` already enforces it on rows, and a graph node
    is a registered domain, so node splits inherit it for free - but "inherits
    for free" is exactly the kind of claim that should be measured, not asserted.
    This reads the authoritative domain->split table and reports the number of
    disagreeing nodes.
    """
    result: dict[str, Any] = {
        "domain_split_csv": str(domain_split_csv),
        "checked": False,
        "domain_nodes_checked": 0,
        "violations": [],
        "n_violations": None,
    }
    if not domain_split_csv.is_file():
        result["reason"] = f"{domain_split_csv} not found"
        return result

    code = {"train": SPLIT_TRAIN, "val": SPLIT_VAL, "test": SPLIT_TEST}
    expected: dict[str, int] = {}
    with domain_split_csv.open("r", encoding="utf-8") as fh:
        next(fh, None)
        for line in fh:
            parts = line.strip().split(",")
            if len(parts) != 2:
                continue
            expected[parts[0].strip().lower()] = code.get(parts[1].strip().lower(), -1)

    violations: list[dict[str, Any]] = []
    checked = 0
    for domain, idx in spec.domain_to_index.items():
        want = expected.get(domain)
        if want is None:
            continue
        checked += 1
        got = int(spec.node_split[idx])
        if got != want:
            violations.append({"domain": domain, "expected": want, "actual": got})

    result.update(
        {
            "checked": True,
            "domain_nodes_checked": checked,
            "violations": violations[:20],
            "n_violations": len(violations),
        }
    )
    return result


# ---------------------------------------------------------------------------
# Tensors / scaling
# ---------------------------------------------------------------------------
def standardise(
    x: np.ndarray, train_rows: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Z-score node features using **training nodes only**.

    Fitting the scaler on the whole graph would let validation and test node
    statistics influence the model - a quiet form of leakage that is easy to
    miss because the features contain no labels.
    """
    if train_rows.size == 0:
        mean = np.zeros(x.shape[1], dtype=np.float64)
        std = np.ones(x.shape[1], dtype=np.float64)
    else:
        mean = x[train_rows].mean(axis=0)
        std = x[train_rows].std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return ((x - mean) / std).astype(np.float32), mean.astype(np.float32), std.astype(np.float32)


def to_tensors(spec: GraphSpec, x: np.ndarray) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    edge_index = torch.as_tensor(spec.edge_index, dtype=torch.long)
    edge_type = torch.as_tensor(spec.edge_type, dtype=torch.long)
    xt = torch.as_tensor(x, dtype=torch.float32)
    yt = torch.as_tensor(np.clip(spec.y, 0, None), dtype=torch.float32)
    return edge_index, edge_type, xt, yt


def _masked_mean_loss(criterion: nn.Module, logits: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> float:
    if int(mask.sum()) == 0:
        return float("nan")
    with torch.no_grad():
        return float(criterion(logits[mask], y[mask]).item())


@torch.no_grad()
def node_logits(model: PhishingGCN, edge_index, edge_type, x) -> np.ndarray:
    model.eval()
    return model.forward(edge_index, x, edge_type).logit.detach().cpu().numpy()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def run_training(
    cfg: dict,
    config_dir: Path,
    seed: int = 42,
    epochs_override: int | None = None,
    max_nodes_override: int | None = None,
    tag: str = "gnn",
    save_graph_to: Path | None = None,
) -> dict[str, Any]:
    """Build the graph, train, evaluate and persist. Returns the metrics dict."""
    set_seed(seed)
    paths = resolve_paths(cfg, config_dir)
    gcfg = {**CFG_DEFAULTS, **(cfg.get(CFG_SECTION) or {})}
    max_nodes = max_nodes_override if max_nodes_override is not None else int(gcfg["max_nodes"])

    log.info(
        "training_start", tag=tag, device="cpu", seed=seed, max_nodes=max_nodes,
        splits_dir=str(paths["splits_dir"]), config=str(config_dir),
        note="cpu-only by construction; no CUDA code path exists here",
    )

    splits = load_splits(paths["splits_dir"])
    log.info(
        "split_sizes_available",
        **{k: len(v) for k, v in splits.items()},
        **{f"{k}_phishing": int(v.labels.sum()) for k, v in splits.items()},
    )

    urls, labels, split_names, used = sample_splits(splits, max_nodes, seed)
    log.info(
        "sampled_urls",
        **{f"{k}_urls": v for k, v in used.items()},
        total_urls=len(urls),
        capped=max_nodes is not None and len(urls) < sum(len(splits[s]) for s in splits),
    )

    spec = build_graph(
        urls,
        labels,
        splits=split_names,
        max_tld_degree=int(gcfg["max_tld_degree"]),
        max_shared_degree=int(gcfg["max_shared_degree"]),
    )

    verification = verify_split_grouping(spec, paths["splits_dir"] / "domain_split.csv")
    if verification["n_violations"]:
        log.error("split_grouping_violated", n_violations=verification["n_violations"],
                  examples=verification["violations"][:5])
    else:
        log.info("split_grouping_verified", checked=verification["domain_nodes_checked"],
                 n_violations=0)

    infra_edges = (spec.edge_type == EDGE_DOMAIN_IP) | (spec.edge_type == EDGE_DOMAIN_SUBNET)
    touched_by_infra = len(set(spec.edge_index[1][infra_edges].tolist())) if bool(infra_edges.any()) else 0
    honest = {
        "ip_nodes": spec.num_ip_nodes,
        "subnet_nodes": spec.num_subnet_nodes,
        "infrastructure_edges": int(infra_edges.sum()),
        "nodes_with_an_infrastructure_edge": touched_by_infra,
        "fraction_of_nodes_with_infrastructure_edge": (
            round(touched_by_infra / max(spec.num_nodes, 1), 6)
        ),
        "structural_domain_domain_edges": spec.num_edges - int(infra_edges.sum()),
    }
    log.info("graph_ready", nodes=spec.num_nodes, edges=spec.num_edges,
             edge_types=spec.edge_type_counts(), **honest)

    train_mask = torch.as_tensor(spec.labelled_split_mask(SPLIT_TRAIN))
    val_mask = torch.as_tensor(spec.labelled_split_mask(SPLIT_VAL))
    test_mask = torch.as_tensor(spec.labelled_split_mask(SPLIT_TEST))
    for name, m in (("train", train_mask), ("val", val_mask), ("test", test_mask)):
        if int(m.sum()) == 0:
            raise RuntimeError(f"no labelled {name} nodes after sampling; raise --max-nodes")

    y_np = np.clip(spec.y, 0, None)
    n_train, n_val, n_test = int(train_mask.sum()), int(val_mask.sum()), int(test_mask.sum())
    xs, scaler_mean, scaler_std = standardise(spec.x, np.flatnonzero(train_mask.numpy()))
    edge_index, edge_type, x, y = to_tensors(spec, xs)
    y_t = torch.as_tensor(y_np, dtype=torch.float32)

    model = PhishingGCN(
        in_dim=xs.shape[1],
        hidden_dim=int(gcfg["hidden_dim"]),
        n_layers=int(gcfg["n_layers"]),
        n_edge_types=N_EDGE_TYPES,
        dropout=float(gcfg["dropout"]),
        head_dropout=float(gcfg["head_dropout"]),
        input_dropout=float(gcfg["input_dropout"]),
    ).to("cpu")
    n_params = sum(p.numel() for p in model.parameters())

    train_labels = y_np[train_mask.numpy()]
    n_pos, n_neg = float((train_labels == 1).sum()), float((train_labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        raise RuntimeError(
            f"training nodes must contain both classes; got {n_pos} phishing / {n_neg} benign. "
            "Raise --max-nodes."
        )
    criterion = WeightedBCE(pos_weight=n_neg / n_pos)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(gcfg["lr"]),
        weight_decay=float(gcfg["weight_decay"]),
    )

    epochs = int(epochs_override or gcfg["epochs"])
    patience = int(gcfg["early_stopping_patience"])
    state = TrainState()
    ckpt_dir = paths["checkpoints_dir"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "graph_model.pt"
    t0 = time.perf_counter()

    for epoch in range(epochs):
        ep_start = time.perf_counter()
        model.train()
        optimizer.zero_grad()
        out = model(edge_index, x, edge_type)
        loss = criterion(out.logit[train_mask], y_t[train_mask])
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(gcfg["grad_clip"]))
        optimizer.step()

        logits = node_logits(model, edge_index, edge_type, x)
        logits_t = torch.as_tensor(logits)
        val_loss = _masked_mean_loss(criterion, logits_t, y_t, val_mask)
        val_probs = 1.0 / (1.0 + np.exp(-logits[val_mask.numpy()]))
        val_metrics = compute_binary_metrics(
            y_np[val_mask.numpy()], val_probs,
            recall_at_fpr_points=cfg.get("evaluation", {}).get("recall_at_fpr_points", (0.01, 0.05, 0.1)),
        )

        improved = val_loss < state.best_val_loss
        record = {
            "epoch": epoch,
            "train_loss": round(float(loss.item()), 6),
            "val_loss": None if np.isnan(val_loss) else round(val_loss, 6),
            "val_accuracy": round(val_metrics.accuracy, 6),
            "val_f1": round(val_metrics.f1, 6),
            "val_roc_auc": None if np.isnan(val_metrics.roc_auc) else round(val_metrics.roc_auc, 6),
            "val_mcc": round(val_metrics.mcc, 6),
            "seconds": round(time.perf_counter() - ep_start, 3),
            "is_best": improved,
        }
        state.history.append(record)
        log.info("epoch_complete", **{k: v for k, v in record.items() if k != "epoch"}, epoch=epoch)

        if improved:
            state.best_val_loss = val_loss
            state.best_epoch = epoch
            state.epochs_without_improvement = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "model_config": model.config(),
                    "feature_names": list(GRAPH_FEATURE_NAMES),
                    "scaler_mean": scaler_mean,
                    "scaler_std": scaler_std,
                    "n_edge_types": N_EDGE_TYPES,
                    "edge_type_names": EDGE_TYPE_NAMES,
                    "train_state": {
                        "best_val_loss": state.best_val_loss,
                        "best_epoch": state.best_epoch,
                        "history": state.history,
                    },
                    "config": cfg,
                    "seed": seed,
                    "tag": tag,
                    "limitation": LIMITATION,
                },
                ckpt_path,
            )
        else:
            state.epochs_without_improvement += 1
            if state.epochs_without_improvement >= patience:
                log.info("early_stopping", epoch=epoch, best_epoch=state.best_epoch,
                         best_val_loss=state.best_val_loss)
                break

    if ckpt_path.is_file():
        blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(blob["model_state"])
    total_seconds = time.perf_counter() - t0

    # --- test ----------------------------------------------------------
    logits = node_logits(model, edge_index, edge_type, x)
    probs = 1.0 / (1.0 + np.exp(-logits))
    test_y = y_np[test_mask.numpy()]
    test_p = probs[test_mask.numpy()]
    test_metrics = compute_binary_metrics(
        test_y, test_p,
        recall_at_fpr_points=cfg.get("evaluation", {}).get("recall_at_fpr_points", (0.01, 0.05, 0.1)),
    )

    # --- majority-class baseline ---------------------------------------
    # Predicting the more frequent *training node* class for every test node is
    # the only baseline that needs no graph at all. Without it, an accuracy of
    # 0.60 on an imbalanced node set looks like a result.
    majority = int(train_labels.mean() >= 0.5)
    baseline_metrics = compute_binary_metrics(
        test_y, np.full(test_y.shape, float(majority)),
        recall_at_fpr_points=cfg.get("evaluation", {}).get("recall_at_fpr_points", (0.01, 0.05, 0.1)),
    )

    graph_path: Path | None = None
    if save_graph_to is not None:
        graph_path = save_graph(spec, save_graph_to, extra_meta={"seed": seed, "tag": tag})

    metrics: dict[str, Any] = {
        "run": {
            "tag": tag,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "seed": seed,
            "device": "cpu",
            "torch_version": torch.__version__,
            "config_file": str(config_dir),
            "profile": cfg.get("profile"),
            "train_seconds": round(total_seconds, 2),
            "n_parameters": int(n_params),
            "graph_config": gcfg,
        },
        "data": {
            "splits_dir": str(paths["splits_dir"]),
            "urls_available": {k: len(v) for k, v in splits.items()},
            "urls_used": used,
            "max_nodes_cap": max_nodes,
            "subsampled": len(urls) < sum(len(splits[s]) for s in splits),
            "nodes_total": spec.num_nodes,
            "nodes_domain": spec.num_domain_nodes,
            "nodes_labelled_train_val_test": [n_train, n_val, n_test],
            "edges_total": spec.num_edges,
            "edge_type_counts": spec.edge_type_counts(),
            "label_conflicts": int(spec.label_conflicts),
        },
        "split_grouping": verification,
        "graph_honesty": {**honest, "limitation": LIMITATION},
        "training": {
            "epochs_requested": epochs,
            "epochs_completed": len(state.history),
            "best_epoch": state.best_epoch,
            "best_val_loss": state.best_val_loss,
            "early_stopped": len(state.history) < epochs,
            "pos_weight": float(n_neg / n_pos),
            "history": state.history,
        },
        "test_metrics": test_metrics.to_dict(),
        "majority_class_baseline": {
            "predicted_class": majority,
            "train_class_fraction_positive": round(float(train_labels.mean()), 6),
            **baseline_metrics.to_dict(),
        },
        "improvement_over_baseline": {
            key: round(float(getattr(test_metrics, key)) - float(getattr(baseline_metrics, key)), 6)
            for key in ("accuracy", "f1", "mcc", "balanced_accuracy")
        },
        "checkpoint": str(ckpt_path),
        "graph_npz": str(graph_path) if graph_path else None,
        "interpretation": (
            "The GNN beats the majority-class baseline only if accuracy/f1/MCC in "
            "`improvement_over_baseline` are clearly positive. Read them together with "
            "`graph_honesty.limitation`."
        ),
    }

    reports_dir = paths["reports_dir"]
    reports_dir.mkdir(parents=True, exist_ok=True)
    json_path = reports_dir / f"metrics_{tag}.json"
    json_path.write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
    md_path = reports_dir / f"metrics_{tag}.md"
    md_path.write_text(render_markdown(metrics), encoding="utf-8")

    log.info(
        "evaluation_complete",
        tag=tag,
        test_accuracy=round(test_metrics.accuracy, 6),
        test_f1=round(test_metrics.f1, 6),
        test_roc_auc=None if np.isnan(test_metrics.roc_auc) else round(test_metrics.roc_auc, 6),
        test_mcc=round(test_metrics.mcc, 6),
        baseline_accuracy=round(baseline_metrics.accuracy, 6),
        checkpoint=str(ckpt_path),
        metrics_json=str(json_path),
        metrics_md=str(md_path),
    )
    return metrics


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------
def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return "n/a" if not np.isfinite(value) else f"{value:.{digits}f}"
    return str(value)


def render_markdown(m: dict[str, Any]) -> str:
    """Short human-readable companion to ``metrics_gnn.json``."""
    tm = m["test_metrics"]
    bm = m["majority_class_baseline"]
    imp = m["improvement_over_baseline"]
    d = m["data"]
    h = m["graph_honesty"]
    t = m["training"]
    # The baseline predicts one constant class, so its F1/recall are a degenerate
    # artefact - it "wins" recall by calling everything phishing. The verdict is
    # therefore judged on the metrics a constant predictor cannot game:
    # accuracy, balanced accuracy and MCC.
    beats = imp["accuracy"] > 0.02 and imp["mcc"] > 0.05 and imp["balanced_accuracy"] > 0.02
    verdict = (
        "The GCN beats the majority-class baseline on accuracy, balanced accuracy "
        "and MCC. That is a real, if modest, gain: the graph branch adds signal the "
        "constant baseline does not have. It is still far below the URL CharCNN-"
        "BiLSTM branch, and with zero infrastructure edges it is a measurement of "
        "name structure alone, not of shared hosting."
        if beats
        else "The GCN does NOT meaningfully beat the majority-class baseline. Treat "
        "this branch as not-yet-working: with no DNS data the structural edges carry "
        "too little signal to beat 'predict the majority class'."
    )
    baseline_note = (
        f"Test nodes: {tm['n_samples']}. The baseline predicts class "
        f"{bm['predicted_class']} for every node "
        f"(training nodes are {bm['train_class_fraction_positive']:.2%} phishing), so "
        "its recall and F1 are inflated by construction - it buys phishing recall by "
        "calling everything phishing and pays for it with specificity "
        f"{_fmt(bm['specificity'])}. Judge the model on accuracy, balanced accuracy "
        "and MCC, which a constant predictor cannot game."
    )
    rows = [
        "# GNN (domain-IP graph) results",
        "",
        f"Generated {m['run']['generated_at_utc']} - seed {m['run']['seed']}, CPU, "
        f"{m['run']['train_seconds']}s, {m['run']['n_parameters']} parameters.",
        "",
        "## Limitation (read first)",
        "",
        f"> **{h['limitation']}**",
        "",
        "Measured on this run: "
        f"{h['infrastructure_edges']} of {d['edges_total']} edges are infrastructure "
        f"(domain-IP / subnet) and only {h['nodes_with_an_infrastructure_edge']} of "
        f"{d['nodes_total']} nodes ({h['fraction_of_nodes_with_infrastructure_edge']:.2%}) "
        "have any infrastructure edge at all. "
        f"{h['structural_domain_domain_edges']} edges are domain<->domain structure.",
        "",
        "## Test metrics (node level)",
        "",
        "| metric | GCN | majority-class baseline | delta |",
        "| --- | --- | --- | --- |",
        f"| accuracy | {_fmt(tm['accuracy'])} | {_fmt(bm['accuracy'])} | {_fmt(imp['accuracy'])} |",
        f"| balanced accuracy | {_fmt(tm['balanced_accuracy'])} | {_fmt(bm['balanced_accuracy'])} | {_fmt(imp['balanced_accuracy'])} |",
        f"| F1 (phishing) | {_fmt(tm['f1'])} | {_fmt(bm['f1'])} | {_fmt(imp['f1'])} |",
        f"| MCC | {_fmt(tm['mcc'])} | {_fmt(bm['mcc'])} | {_fmt(imp['mcc'])} |",
        f"| recall (phishing) | {_fmt(tm['recall'])} | {_fmt(bm['recall'])} | - |",
        f"| specificity | {_fmt(tm['specificity'])} | {_fmt(bm['specificity'])} | - |",
        f"| ROC-AUC | {_fmt(tm['roc_auc'])} | {_fmt(bm['roc_auc'])} | - |",
        f"| PR-AUC | {_fmt(tm['pr_auc'])} | {_fmt(bm['pr_auc'])} | - |",
        f"| recall @ FPR 0.01 | {_fmt(tm['recall_at_fpr'].get('0.01'))} | "
        f"{_fmt(bm['recall_at_fpr'].get('0.01'))} | - |",
        f"| Brier | {_fmt(tm['brier'])} | {_fmt(bm['brier'])} | - |",
        "",
        baseline_note,
        "",
        "## Split integrity",
        "",
        f"- node-level split derived from `data/splits/` (grouped by registered domain): "
        f"{d['nodes_labelled_train_val_test'][0]} train / "
        f"{d['nodes_labelled_train_val_test'][1]} val / "
        f"{d['nodes_labelled_train_val_test'][2]} test labelled nodes",
        f"- cross-checked against `domain_split.csv`: "
        f"{m['split_grouping'].get('n_violations')} domain(s) disagree "
        f"({m['split_grouping'].get('domain_nodes_checked')} checked)",
        "- a domain in train cannot be a test node, because the node *is* the domain",
        "- graph construction never reads a label (see `app/preprocessing/graph_build.py`)",
        "",
        "## Graph",
        "",
        f"- nodes: {d['nodes_total']} ({d['nodes_domain']} domain, {h['ip_nodes']} IP, "
        f"{h['subnet_nodes']} subnet)",
        f"- edges: {d['edges_total']}",
        f"- edge types: {json.dumps(d['edge_type_counts'])}",
        f"- URLs sampled: {json.dumps(d['urls_used'])} of {json.dumps(d['urls_available'])}",
        "",
        "## Training",
        "",
        f"- epochs run {t['epochs_completed']}/{t['epochs_requested']} "
        f"(early stopped: {t['early_stopped']}), best epoch {t['best_epoch']}",
        f"- selection metric: **validation loss** "
        f"({_fmt(t['best_val_loss'])}), phishing pos_weight {_fmt(t['pos_weight'])}",
        "",
        "## Verdict",
        "",
        verdict,
        "",
    ]
    return "\n".join(rows)


# ---------------------------------------------------------------------------
def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Train the domain-IP phishing GCN.")
    p.add_argument("--config", type=Path, default=BACKEND_ROOT / "configs" / "dev.yaml")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None, help="override the epoch count")
    p.add_argument("--max-nodes", type=int, default=None,
                   help="cap on total URLs fed to the graph builder (CPU practicality)")
    p.add_argument("--tag", default="gnn", help="report name suffix")
    p.add_argument("--no-save-graph", action="store_true",
                   help="skip writing data/graph/*.npz")
    args = p.parse_args(argv)

    configure_logging(fmt="json")
    cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else int(cfg.get("runtime", {}).get("seed", 42))

    graph_to = None if args.no_save_graph else default_graph_path(BACKEND_ROOT)
    metrics = run_training(
        cfg, args.config, seed=seed,
        epochs_override=args.epochs,
        max_nodes_override=args.max_nodes,
        tag=args.tag,
        save_graph_to=graph_to,
    )
    tm = metrics["test_metrics"]
    bm = metrics["majority_class_baseline"]
    print(json.dumps({
        "test_accuracy": tm["accuracy"],
        "test_f1": tm["f1"],
        "test_roc_auc": tm["roc_auc"],
        "test_pr_auc": tm["pr_auc"],
        "test_mcc": tm["mcc"],
        "test_recall": tm["recall"],
        "test_specificity": tm["specificity"],
        "baseline_accuracy": bm["accuracy"],
        "baseline_f1": bm["f1"],
        "baseline_mcc": bm["mcc"],
        "improvement_over_baseline": metrics["improvement_over_baseline"],
        "nodes": metrics["data"]["nodes_total"],
        "edges": metrics["data"]["edges_total"],
        "nodes_with_infrastructure_edge": metrics["graph_honesty"]["nodes_with_an_infrastructure_edge"],
        "metrics_json": str(BACKEND_ROOT / "reports" / f"metrics_{args.tag}.json"),
        "metrics_md": str(BACKEND_ROOT / "reports" / f"metrics_{args.tag}.md"),
    }, indent=2))
    print("\nLIMITATION: " + LIMITATION)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
