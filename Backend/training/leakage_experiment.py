"""P2 leakage experiment - what PhiUSIIL's page-derived columns are worth.

This is an **experiment, not a candidate model**. It answers one question:
does training on PhiUSIIL's engineered columns produce a materially better-looking
result than training on the raw URL string alone?

The answer matters because those columns are computed by *fetching the page*:
``URLSimilarityIndex`` compares the live URL against a reference, ``LineOfCode``,
``LargestLineLength``, ``HasPasswordField``, ``NoOfiFrame`` and friends describe
the rendered document. None of them exist at inference time for a system that is
handed only a URL string. A model trained on them is therefore measuring
collection-time information, not URL-string signal, and any headline number taken
from it would overstate what the shipped system can do.

Protocol, kept deliberately comparable:

* the **same** grouped split and the same row ids as the URL model
* engineered columns only, URL string excluded entirely
* two model families so the result is not an artefact of one architecture:
  a PyTorch MLP (matching the project's modelling stack) and scikit-learn
  logistic regression (a deliberately weak, low-variance reference)
* metrics reported through the same helper as the headline model

Engaged explicitly:
    python training/train_url.py --config configs/dev.yaml --leakage-experiment
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

import numpy as np
import pandas as pd
import torch
from torch import nn

from app.preprocessing.url_dataset import to_label_array
from app.utils.metrics import compute_binary_metrics

#: Columns that describe the raw URL string itself and are therefore legitimately
#: available to a URL model. Everything else in the file is page-derived.
URL_DERIVED_COLUMNS: frozenset[str] = frozenset(
    {
        "URL",
        "URLLength",
        "Domain",
        "DomainLength",
        "IsDomainIP",
        "TLD",
        "TLDLength",
        "NoOfSubDomain",
        "IsHTTPS",
        "NoOfDegitsInURL",
        "DegitRatioInURL",
        "NoOfEqualsInURL",
        "NoOfQMarkInURL",
        "NoOfAmpersandInURL",
        "NoOfOtherSpecialCharsInURL",
        "SpacialCharRatioInURL",
        "NoOfLettersInURL",
        "LetterRatioInURL",
        "URLCharProb",
        "CharContinuationRate",
        "HasObfuscation",
        "NoOfObfuscatedChar",
        "ObfuscationRatio",
    }
)

#: Columns that are known to be *derived from the fetched page*. Listed for the
#: report; membership is not required for the experiment, the split is by
#: exclusion from URL_DERIVED_COLUMNS.
NOTABLE_PAGE_DERIVED: tuple[str, ...] = (
    "URLSimilarityIndex",
    "TLDLegitimateProb",
    "LineOfCode",
    "LargestLineLength",
    "HasTitle",
    "Title",
    "DomainTitleMatchScore",
    "URLTitleMatchScore",
    "HasFavicon",
    "Robots",
    "IsResponsive",
    "NoOfURLRedirect",
    "NoOfSelfRedirect",
    "HasDescription",
    "NoOfPopup",
    "NoOfiFrame",
    "HasExternalFormSubmit",
    "HasSocialNet",
    "HasSubmitButton",
    "HasHiddenFields",
    "HasPasswordField",
    "Bank",
    "Pay",
    "Crypto",
    "HasCopyrightInfo",
    "NoOfImage",
    "NoOfCSS",
    "NoOfJS",
    "NoOfSelfRef",
    "NoOfEmptyRef",
    "NoOfExternalRef",
)


#: Structural columns that are never features. ``label`` is the target itself:
#: the source file also carries a column by that name, and including it would
#: hand the model the answer.
NON_FEATURE_COLUMNS: frozenset[str] = frozenset({"row_id", "url", "label_raw", "URL", "label"})


def load_engineered_for_rows(dataset_csv: Path, dedup_csv: Path) -> pd.DataFrame:
    """Recover the page-derived columns for exactly the deduplicated rows.

    ``dedup.csv`` guarantees ``url`` is unique, so joining the original file on
    ``url`` (first occurrence, matching ``drop_duplicates(keep='first')``) is an
    exact, verifiable correspondence - and avoids relying on positional indices
    into a file that has since been re-read.
    """
    dedup = pd.read_csv(dedup_csv, dtype={"url": str, "label_raw": str})
    if not dedup["url"].is_unique:
        raise RuntimeError("dedup.csv contains duplicate URLs; the join would be ambiguous")

    wanted = set(dedup["url"])
    header = pd.read_csv(dataset_csv, nrows=0)
    # 'URL' is needed as the join key even though it is itself URL-derived.
    usecols = [c for c in header.columns if c not in URL_DERIVED_COLUMNS or c == "URL"]
    missing_keys = {"URL", "label"} - set(usecols)
    if missing_keys:
        raise RuntimeError(f"dataset is missing required column(s): {sorted(missing_keys)}")

    kept: list[pd.DataFrame] = []
    for chunk in pd.read_csv(dataset_csv, usecols=usecols, chunksize=200_000, dtype=str):
        sub = chunk[chunk["URL"].isin(wanted)]
        if not sub.empty:
            kept.append(sub.drop_duplicates(subset=["URL"], keep="first"))
    if not kept:
        raise RuntimeError("no rows of dedup.csv were found in the dataset")

    src = pd.concat(kept, ignore_index=True)
    merged = dedup[["row_id", "url", "label_raw"]].merge(
        src, left_on="url", right_on="URL", how="left"
    )
    missing = int(merged["URL"].isna().sum())
    if missing:
        raise RuntimeError(f"{missing} deduplicated rows had no match in the dataset file")
    return merged


def _to_numeric(df: pd.DataFrame, columns: list[str]) -> np.ndarray:
    """Coerce engineered columns to float, with explicit sentinels.

    ``Title`` is a free-text column, so it is represented here by two
    interpretable scalars (length, digit count) rather than fed as a string: a
    linear/MLP input must be numeric, and inventing an encoding for it would add
    noise without addressing the leakage question.
    """
    out: list[np.ndarray] = []
    for col in columns:
        if col == "Title":
            s = df[col].fillna("").astype(str)
            out.append(s.str.len().to_numpy(dtype=np.float32)[:, None])
            out.append(s.str.count(r"\d").to_numpy(dtype=np.float32)[:, None])
            continue
        numeric = pd.to_numeric(df[col], errors="coerce")
        # NaN -> -1 so the model sees "missing" as its own value, and so
        # imputation cannot quietly become a label shortcut.
        out.append(numeric.fillna(-1.0).to_numpy(dtype=np.float32)[:, None])
    return np.concatenate(out, axis=1)


class EngineeredMLP(nn.Module):
    """Small MLP over the engineered feature vector."""

    def __init__(self, n_in: int, hidden: int = 128, dropout: float = 0.3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def _train_mlp(
    x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, seed: int, epochs: int
) -> np.ndarray:
    torch.manual_seed(seed)
    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True)
    std[std < 1e-8] = 1.0
    xtr = torch.tensor((x_train - mean) / std, dtype=torch.float32)
    xte = torch.tensor((x_test - mean) / std, dtype=torch.float32)
    ytr = torch.tensor(y_train, dtype=torch.float32)

    n_pos = float(y_train.sum())
    n_neg = float(len(y_train) - n_pos)
    pos_weight = torch.tensor(n_neg / max(n_pos, 1.0))

    model = EngineeredMLP(x_train.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    ds = torch.utils.data.TensorDataset(xtr, ytr)
    loader = torch.utils.data.DataLoader(ds, batch_size=256, shuffle=True)
    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(xte)).numpy()


def run_leakage_experiment(cfg: dict, config_path: Path, seed: int = 42, epochs: int | None = None) -> int:
    from app.core.logging_config import get_logger
    from train_url import resolve_paths

    log = get_logger("leakage")
    paths = resolve_paths(cfg, config_path)
    splits_dir = paths["splits_dir"]
    dataset_csv = paths["dataset_csv"]

    merged = load_engineered_for_rows(dataset_csv, splits_dir / "dedup.csv")
    by_id = merged.set_index("row_id")

    index = {
        name: [int(x) for x in (splits_dir / f"{name}.txt").read_text(encoding="utf-8").split()]
        for name in ("train", "val", "test")
    }

    engineered_cols = [
        c for c in merged.columns if c not in NON_FEATURE_COLUMNS
    ]
    if not engineered_cols:
        raise RuntimeError("no engineered columns found; the exclusion set is wrong")

    rows = {name: by_id.loc[idx] for name, idx in index.items()}
    X = {name: _to_numeric(frame, engineered_cols) for name, frame in rows.items()}
    y = {name: to_label_array(frame["label_raw"].tolist()) for name, frame in rows.items()}

    # Guard: a feature that reproduces the label exactly is a direct leak and must
    # abort the experiment rather than produce a meaningless AUC of 1.0.
    for i, col in enumerate(engineered_cols):
        col_vals = X["test"][:, i]
        if np.array_equal((col_vals > 0).astype(int), y["test"]):
            raise RuntimeError(
                f"engineered column {col!r} exactly reproduces the label; refusing to "
                "report an experiment that would be trivially 'solved'"
            )

    log.info(
        "leakage_data_ready",
        n_engineered_columns=len(engineered_cols),
        n_train=len(X["train"]), n_val=len(X["val"]), n_test=len(X["test"]),
    )

    results: dict[str, Any] = {}
    n_ep = epochs or int(cfg["url_model"].get("epochs", 5))

    # --- MLP -------------------------------------------------------------
    mlp_prob = _train_mlp(X["train"], y["train"], X["test"], seed, n_ep)
    mlp_m = compute_binary_metrics(y["test"], mlp_prob)
    results["engineered_mlp"] = {
        "architecture": f"MLP(128, 64) over {len(engineered_cols)} engineered columns",
        **mlp_m.to_dict(),
    }

    # --- logistic regression reference ------------------------------------
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(X["train"])
    lr = LogisticRegression(max_iter=2000, class_weight="balanced")
    lr.fit(scaler.transform(X["train"]), y["train"])
    lr_prob = lr.predict_proba(scaler.transform(X["test"]))[:, 1]
    lr_m = compute_binary_metrics(y["test"], lr_prob)
    results["engineered_logistic_regression"] = {
        "architecture": f"LogisticRegression over {len(engineered_cols)} engineered columns",
        **lr_m.to_dict(),
    }

    # --- per-column leakage probe ----------------------------------------
    # Single-feature AUCs show which individual columns carry the most label
    # information. A column that alone nearly solves the task is a red flag for
    # collection-time leakage.
    from sklearn.metrics import roc_auc_score

    column_auc: dict[str, float] = {}
    for i, col in enumerate(engineered_cols):
        try:
            col_vals = X["test"][:, i]
            if np.allclose(col_vals, col_vals[0]):
                column_auc[col] = float("nan")
            else:
                column_auc[col] = float(roc_auc_score(y["test"], col_vals))
        except Exception:  # noqa: BLE001
            column_auc[col] = float("nan")
    ranked = sorted(
        ((k, v) for k, v in column_auc.items() if np.isfinite(v)),
        key=lambda kv: -abs(kv[1] - 0.5),
    )

    url_only_path = paths["reports_dir"] / "metrics_url.json"
    comparison: dict[str, Any] = {}
    if url_only_path.is_file():
        url_metrics = json.loads(url_only_path.read_text(encoding="utf-8"))
        u = url_metrics["test_metrics_calibrated"]
        comparison = {
            "url_only_model": {
                "recall": u["recall"],
                "precision": u["precision"],
                "f1": u["f1"],
                "roc_auc": u["roc_auc"],
                "pr_auc": u["pr_auc"],
            },
            "engineered_mlp": {
                "recall": results["engineered_mlp"]["recall"],
                "precision": results["engineered_mlp"]["precision"],
                "f1": results["engineered_mlp"]["f1"],
                "roc_auc": results["engineered_mlp"]["roc_auc"],
                "pr_auc": results["engineered_mlp"]["pr_auc"],
            },
            "delta_recall_mlp_minus_url": (
                results["engineered_mlp"]["recall"] - u["recall"]
            ),
        }

    payload = {
        "experiment": "PhiUSIIL page-derived engineered columns vs URL string only",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "epochs": n_ep,
        "note": (
            "This is a LEAKAGE EXPERIMENT. The engineered columns are computed by "
            "fetching the page, so they are unavailable to a system that receives only "
            "a URL string. Metrics here must not be quoted as achievable system "
            "performance; they quantify what collection-time information is worth."
        ),
        "dataset": {
            "csv": str(dataset_csv),
            "splits_dir": str(splits_dir),
            "n_engineered_columns": len(engineered_cols),
            "engineered_column_names": engineered_cols,
            "url_derived_columns_excluded": sorted(URL_DERIVED_COLUMNS),
            "notable_page_derived_present": [
                c for c in NOTABLE_PAGE_DERIVED if c in engineered_cols
            ],
            "rows": {k: int(len(v)) for k, v in X.items()},
        },
        "results": results,
        "single_column_auc_on_test": {k: v for k, v in ranked},
        "strongest_single_columns": [
            {"column": k, "auc": v} for k, v in ranked[:12]
        ],
        "comparison_to_url_only_model": comparison,
    }

    out_path = paths["reports_dir"] / "leakage_experiment.json"
    out_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    md = _render_markdown(payload)
    md_path = paths["reports_dir"] / "leakage_experiment.md"
    md_path.write_text(md, encoding="utf-8")

    print(json.dumps({
        "engineered_mlp": {
            k: results["engineered_mlp"][k]
            for k in ("accuracy", "precision", "recall", "specificity", "f1", "roc_auc", "pr_auc", "mcc")
        },
        "engineered_logistic_regression": {
            k: results["engineered_logistic_regression"][k]
            for k in ("accuracy", "precision", "recall", "specificity", "f1", "roc_auc", "pr_auc", "mcc")
        },
        "comparison": comparison,
        "strongest_single_columns": payload["strongest_single_columns"][:5],
        "report": str(md_path),
    }, indent=2))
    return 0


def _render_markdown(p: dict[str, Any]) -> str:
    lines: list[str] = []
    ap = lines.append
    ap("# Leakage Experiment: PhiUSIIL page-derived columns vs URL string only")
    ap("")
    ap(f"Generated {p['generated_at_utc']} by `training/leakage_experiment.py`.")
    ap("")
    ap("> " + p["note"].replace("\n", " "))
    ap("")

    ap("## What was trained")
    ap("")
    ap(f"- **Engineered-only models**: {p['dataset']['n_engineered_columns']} columns, "
       "excluding every column derivable from the URL string itself.")
    ap("- **URL-only model**: the headline `metrics_url.json` run, same split, same row ids.")
    ap("")
    ap("### Columns deliberately excluded (URL-derived, legitimately available)")
    ap("")
    ap("```")
    ap(", ".join(p["dataset"]["url_derived_columns_excluded"]))
    ap("```")
    ap("")
    ap("### Notable page-derived columns included in the experiment")
    ap("")
    ap("```")
    ap(", ".join(p["dataset"]["notable_page_derived_present"]))
    ap("```")
    ap("")

    ap("## Test-split results")
    ap("")
    ap("| Model | Accuracy | Precision | Phishing recall | Specificity | F1 | ROC-AUC | PR-AUC | MCC | Brier |")
    ap("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for key, r in p["results"].items():
        ap(
            f"| {key} | {r['accuracy']:.4f} | {r['precision']:.4f} | {r['recall']:.4f} | "
            f"{r['specificity']:.4f} | {r['f1']:.4f} | {r['roc_auc']:.4f} | "
            f"{r['pr_auc']:.4f} | {r['mcc']:.4f} | {r['brier']:.4f} |"
        )
    if p["comparison_to_url_only_model"]:
        ap("")
        u = p["comparison_to_url_only_model"]["url_only_model"]
        e = p["comparison_to_url_only_model"]["engineered_mlp"]
        ap("| Comparison vs URL-only model | Phishing recall | Precision | F1 | ROC-AUC |")
        ap("| --- | ---: | ---: | ---: | ---: |")
        ap(f"| URL-only (shipped) | {u['recall']:.4f} | {u['precision']:.4f} | {u['f1']:.4f} | {u['roc_auc']:.4f} |")
        ap(f"| Engineered MLP | {e['recall']:.4f} | {e['precision']:.4f} | {e['f1']:.4f} | {e['roc_auc']:.4f} |")
        d = p["comparison_to_url_only_model"]["delta_recall_mlp_minus_url"]
        ap(f"| **Delta** | **{d:+.4f}** | | | |")
    ap("")

    ap("## Strongest single columns (test-split AUC, |AUC - 0.5| ranked)")
    ap("")
    ap("A single column that alone nearly separates the classes is direct evidence")
    ap("of collection-time leakage.")
    ap("")
    ap("| Column | AUC |")
    ap("| --- | ---: |")
    for item in p["strongest_single_columns"]:
        ap(f"| `{item['column']}` | {item['auc']:.4f} |")
    ap("")
    ap("### A caveat on the top-ranked column")
    ap("")
    ap("`URLSimilarityIndex` is PhiUSIIL's share of *near-duplicate URLs in the corpus*.")
    ap("It is built from URL strings alone, so it is not page-derived in the strict")
    ap("sense, but it cannot be computed for a previously unseen URL without the")
    ap("reference corpus it was counted against. It is grouped with the engineered")
    ap("columns because a deployed detector cannot produce it for an arbitrary input,")
    ap("and because a value of exactly 0 for a novel URL versus a populated value for")
    ap("a URL already present in training is a dataset artifact rather than a property")
    ap("of the URL. It is reported here as measured; it is not usable by the shipped")
    ap("model. Every remaining row in the table above is a genuine page-content")
    ap("measurement.")
    ap("")

    ap("## Conclusion")
    ap("")
    if p["comparison_to_url_only_model"]:
        d = p["comparison_to_url_only_model"]["delta_recall_mlp_minus_url"]
        if d > 0.02:
            ap(f"The engineered-column model reaches **{d:+.4f}** higher phishing recall than")
            ap("the URL-only model on the identical test split. Those columns are not")
            ap("derivable from a URL string, so this gap is **not deployable**: it")
            ap("reflects information captured while crawling the page. Training the")
            ap("headline model on them would overstate achievable performance and")
            ap("would fail outright on any live URL that has not been crawled.")
        elif d < -0.02:
            ap(f"The engineered-column model is actually **{d:+.4f}** *worse* than the")
            ap("URL-only model on the identical split. Even with the page-derived")
            ap("features in hand, the raw URL string is not the weaker signal. This is")
            ap("reported as measured, without reinterpretation.")
        else:
            ap(f"The two models are within {abs(d):.4f} phishing recall of each other, so")
            ap("the page-derived columns add no usable signal beyond the URL string on")
            ap("this split. That is the *best* outcome for the integrity of the project:")
            ap("the leak, while present in the data, does not create a fake headline gain.")
    ap("")
    ap("The shipped URL model uses only the raw URL string plus the separately")
    ap("documented handcrafted branch computed from that string alone. The engineered")
    ap("columns are used **only** in this experiment.")
    ap("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    import argparse

    from app.core.logging_config import configure_logging
    from train_url import load_config

    backend_root = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser(
        description="Leakage experiment: train on PhiUSIIL page-derived columns."
    )
    p.add_argument("--config", type=Path, default=backend_root / "configs" / "dev.yaml")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None, help="override the config epoch count")
    args = p.parse_args(argv)

    configure_logging(fmt="json")
    cfg = load_config(args.config)
    seed = args.seed if args.seed is not None else int(cfg.get("runtime", {}).get("seed", 42))
    return run_leakage_experiment(cfg, args.config, seed=seed, epochs=args.epochs)


if __name__ == "__main__":
    raise SystemExit(main())
