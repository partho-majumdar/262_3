"""Build the project's Jupyter notebooks.

Kept as a generator rather than hand-edited JSON so that the shared bootstrap
cell, styling and section skeleton stay consistent across all six notebooks.
Run: python notebooks/_build_notebooks.py
"""

from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent
BACKEND = HERE.parent

nb = nbf.v4.new_notebook()


_CELL_ID = 0


def _next_id():
    global _CELL_ID
    _CELL_ID += 1
    return f"cell-{_CELL_ID:03d}"


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip("\n"), id=_next_id())


def code(text: str, **kw):
    return nbf.v4.new_code_cell(text.strip("\n"), id=_next_id(), **kw)


def save(name: str, cells, title: str):
    bad = [(i, type(c).__name__) for i, c in enumerate(cells) if not hasattr(c, "get")]
    if bad:
        raise TypeError(f"{name}: non-cell entries at {bad}")
    n = nbf.v4.new_notebook(cells=cells)
    n.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.10"},
        "project_title": title,
    }
    path = HERE / name
    nbf.write(n, path)
    print(f"wrote {path.name} ({len(cells)} cells)")


# --------------------------------------------------------------------------
# Shared bootstrap cell reused by every notebook
# --------------------------------------------------------------------------
BOOTSTRAP = f'''
import sys
from pathlib import Path

BACKEND = Path(r"{BACKEND}")
DATASET = BACKEND.parent / "Dataset"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

import json
import numpy as np
import pandas as pd

pd.set_option("display.max_columns", 60)
pd.set_option("display.width", 160)

SEED = 42
np.random.seed(SEED)

print("backend :", BACKEND)
print("dataset :", DATASET)
print("numpy   :", np.__version__)
print("pandas  :", pd.__version__)
'''

STYLE = '''
def show(title, obj):
    print(f"\\n=== {title} ===")
    print(obj)

def hr(title):
    print("\\n" + "=" * 70)
    print(title)
    print("=" * 70)
'''

SAVE_HELPER = '''
NOTEBOOK_DIR = BACKEND / "notebooks"
ARTIFACT_DIR = NOTEBOOK_DIR / "artifacts"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

def save_artifact(name, obj):
    """Persist a JSON-serialisable object and echo where it landed."""
    path = ARTIFACT_DIR / name
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
    print(f"saved -> {path}")
    return path
'''

CAUTION = """
> **Note.** These notebooks *call* the production modules in `app/` and
> `training/`. They do not reimplement the pipeline. The logic under test lives
> in exactly one place, so a notebook can never drift from what the CLI scripts
> and the API actually run.
"""

# ==========================================================================
# 01 - Data loading and inspection
# ==========================================================================
save(
    "01_data_loading_and_inspection.ipynb",
    [
        md(
            """
# 01 — Dataset loading and inspection

**Source:** `Dataset/Phishing_URL_Dataset.csv` (PhiUSIIL), 235,795 rows, 55 columns.
SHA-256 `748ae7c7a677ac4ffc0eeefbee653dcef1d685b134ad2f5376596b92d79438ba`

## Label convention

`0 = Legitimate`, `1 = Phishing`. This is the mapping the whole project uses.
Any notebook or report claiming the reverse is wrong.
"""
        ),
        md("## 1.1 Environment"),
        code(BOOTSTRAP),
        code(STYLE),
        md(
            """
## 1.2 Verify the dataset file

Hash first. If the digest does not match, every downstream number is suspect and
you should stop rather than train on an unknown file.
"""
        ),
        code(
            '''
import hashlib

DATASET_CSV = DATASET / "Phishing_URL_Dataset.csv"
EXPECTED_SHA256 = "748ae7c7a677ac4ffc0eeefbee653dcef1d685b134ad2f5376596b92d79438ba"

show("dataset file", {
    "path": str(DATASET_CSV),
    "exists": DATASET_CSV.exists(),
    "size_bytes": DATASET_CSV.stat().st_size if DATASET_CSV.exists() else None,
})

h = hashlib.sha256()
with DATASET_CSV.open("rb") as fh:
    for chunk in iter(lambda: fh.read(1 << 20), b""):
        h.update(chunk)
digest = h.hexdigest()

show("sha256", {"actual": digest, "expected": EXPECTED_SHA256, "match": digest == EXPECTED_SHA256})
assert digest == EXPECTED_SHA256, "Dataset hash mismatch — stop and investigate."
'''
        ),
        md(
            """
## 1.3 Column detection

`training/inspect_dataset.py` picks the URL and label columns by *content*, not by
name: the URL column must parse as an absolute URL at a rate of >= 0.5. This is why
the schemeless second dataset in `Dataset/` is rejected outright.
"""
        ),
        code(
            '''
from training.inspect_dataset import detect_columns

url_col, label_col, evidence = detect_columns(DATASET_CSV)
show("detected columns", {"url_column": url_col, "label_column": label_col})
show("evidence", evidence)
'''
        ),
        md("## 1.4 Load and inspect"),
        code(
            '''
df = pd.read_csv(DATASET_CSV, usecols=[url_col, label_col], dtype=str)

show("shape", df.shape)
show("label value counts", df[label_col].value_counts(dropna=False))
show("nulls", df.isna().sum().to_dict())
show("duplicate URLs", int(df[url_col].duplicated().sum()))
show("head", df.head())
'''
        ),
        md(
            """
## 1.5 Label semantics — check this, do not assume

Map raw labels to integers with the project's own function rather than a hand-rolled
`astype(int)`. It raises on anything it cannot interpret instead of silently guessing.
"""
        ),
        code(
            '''
from app.preprocessing.url_dataset import to_label_array

y = to_label_array(df[label_col].tolist())
hr("label mapping")
print("1 = phishing, 0 = legitimate")
show("counts", pd.Series(y).value_counts().sort_index().to_dict())
show("phishing fraction", round(float(y.mean()), 6))

df = df.assign(label=y)
show("head with mapped labels", df.head())
'''
        ),
        md("## 1.6 Host and registered-domain structure"),
        code(
            '''
from app.preprocessing.url_features import registered_domain_of
from app.preprocessing.url_preprocessing import host_of

hosts = df[url_col].map(host_of)
regs = df[url_col].map(registered_domain_of)

show("unparseable hosts", int((hosts == "").sum()))
show("unique hosts", int(hosts.nunique()))
show("unique registered domains", int(regs.nunique()))

# A host carrying both labels is a label-noise signal in PhiUSIIL.
lab = pd.DataFrame({"host": hosts, "label": df["label"]})
conflict = lab.groupby("host")["label"].nunique()
show("hosts with BOTH labels", int((conflict > 1).sum()))
'''
        ),
        md("## 1.7 Class balance by registered domain"),
        code(
            '''
per_reg = pd.DataFrame({"registered_domain": regs, "label": df["label"]}) \\
    .groupby("registered_domain")["label"].agg(["size", "mean"])

hr("registered-domain level")
show("domains", int(len(per_reg)))
show("single-row domains", int((per_reg["size"] == 1).sum()))
show("pure-phishing domains", int((per_reg["mean"] == 1.0).sum()))
show("pure-legitimate domains", int((per_reg["mean"] == 0.0).sum()))

# The reason the split groups by domain: random row-level splitting lets the
# same brand appear in train and test, which inflates scores.
show("top domains by row count", per_reg.sort_values("size", ascending=False).head(10))
'''
        ),
        md(CAUTION),
        md(
            """
## Next

`02_url_preprocessing.ipynb` — tokenizer, 30 handcrafted features, scaler.
"""
        ),
    ],
    "01 — Data loading and inspection",
)

# ==========================================================================
# 02 - URL preprocessing
# ==========================================================================
save(
    "02_url_preprocessing.ipynb",
    [
        md(
            """
# 02 — URL preprocessing

Three pieces, all in `app/preprocessing/`:

| Component | Class | Output |
|---|---|---|
| Character tokenizer | `CharTokenizer` | `(ids, mask)` of shape `(B, 256)` |
| Handcrafted features | `extract_batch` | `(B, 30)` float32 |
| Feature scaler | `FeatureScaler` | z-scored `(B, 30)` |

**Order matters:** fit the tokenizer and scaler on the **train split only**, then
transform val/test. Fitting on everything leaks test statistics into training.
"""
        ),
        code(BOOTSTRAP),
        code(STYLE + SAVE_HELPER),
        md("## 2.1 Normalisation"),
        code(
            '''
from app.preprocessing.url_preprocessing import normalize_url, host_of

samples = [
    "HTTP://WWW.Example.COM/Path?a=1&b=2",
    "  http://user:pw@example.com/secure  ",
    "http://example.com/%2e%2e/secret",
]
for s in samples:
    print(f"{s!r}\\n  -> {normalize_url(s)!r}  host={host_of(s)!r}")
'''
        ),
        md(
            """
## 2.2 Splits

`load_splits` reads `data/splits/dedup.csv` plus the three index files produced by
`training/make_splits.py`. If these are missing, run `make_splits.py` first
(see notebook 06 for the in-notebook equivalent).
"""
        ),
        code(
            '''
from pathlib import Path
from app.preprocessing.url_dataset import load_splits

SPLITS_DIR = BACKEND / "data" / "splits"
show("splits dir", {"path": str(SPLITS_DIR), "exists": SPLITS_DIR.exists()})

splits = load_splits(SPLITS_DIR)
for name, sp in splits.items():
    print(f"{name:>5}: n={len(sp):>7,}  phishing_rate={sp.labels.mean():.4f}")
'''
        ),
        md(
            """
## 2.3 Tokenizer — fit on train only

`CharTokenizer` maps characters to ids with `min_count` filtering. `PAD` is always id 0,
which is why the model embedding uses `padding_idx=0`.
"""
        ),
        code(
            '''
from app.preprocessing.url_preprocessing import CharTokenizer, suggest_max_length

MAX_LEN = 256

train_urls = splits["train"].urls
suggested = suggest_max_length(train_urls)
print("suggest_max_length(train):", suggested, " (config uses 256)")

tok = CharTokenizer(max_length=MAX_LEN, min_count=5)
tok.fit(train_urls)

show("vocab", {
    "vocab_size": tok.vocab_size,
    "pad_id": tok.pad_id,
    "unk_id": tok.unk_id,
    "max_length": MAX_LEN,
})

ids, mask = tok.encode("http://secure-login.paypa1.com/verify")
print("ids :", ids[:40])
print("mask:", "".join(str(x) for x in mask[:40]))
assert len(ids) == MAX_LEN and len(mask) == MAX_LEN
'''
        ),
        md("### Truncation rate — is 256 chars actually enough?"),
        code(
            '''
lens = np.array([len(normalize_url(u)) for u in train_urls[:50000]])
over = int((lens > MAX_LEN).sum())
print(f"URLs exceeding {MAX_LEN} chars: {over}/{len(lens)} ({100*over/len(lens):.2f}%)")
show("length percentiles", {f"p{p}": int(np.percentile(lens, p)) for p in (50, 90, 95, 99, 99.9, 100)})
'''
        ),
        md("## 2.4 Handcrafted features — 30 of them"),
        code(
            '''
from app.preprocessing.url_features import FEATURE_NAMES, extract_batch, extract_handcrafted_features

print(f"n features = {len(FEATURE_NAMES)}")
for i, n in enumerate(FEATURE_NAMES):
    print(f"  {i:>2}  {n}")
'''
        ),
        code(
            '''
example = "http://secure-login.paypa1-verify.tk/account/update?id=918273645"
feats = extract_handcrafted_features(example)
assert len(feats) == len(FEATURE_NAMES)
show("features for a suspicious URL", dict(zip(FEATURE_NAMES, [round(v, 4) for v in feats])))
'''
        ),
        md("## 2.5 Scaler — fit on train only"),
        code(
            '''
from app.preprocessing.url_features import FeatureScaler, extract_batch

N_TRAIN = 20000
train_subset = splits["train"].urls[:N_TRAIN]

rows = extract_batch(train_subset)
print("raw feature matrix:", np.asarray(rows).shape)

scaler = FeatureScaler()
scaler.fit(rows)

z = np.asarray(scaler.transform(rows[:5]))
show("scaled sample (first 5 rows)", pd.DataFrame(z[:, :8], columns=list(FEATURE_NAMES)[:8]).round(3))

st = scaler.state_dict()
show("scaler state keys", list(st.keys()))
show("feature_names matches module constant", st["feature_names"] == list(FEATURE_NAMES))
'''
        ),
        md(
            """
## 2.6 Guard against leakage

Refit the scaler on train+val and compare to the train-only fit. A large shift means
val statistics differ from train, which is worth knowing before trusting the val curve.
"""
        ),
        code(
            '''
train_val = splits["train"].urls + splits["val"].urls[:20000]
scaler_leaky = FeatureScaler().fit(extract_batch(train_val))

delta = np.abs(np.asarray(scaler_leaky.state_dict()["mean"]) - np.asarray(st["mean"]))
show("mean shift per feature (top 5)", {
    n: round(float(d), 6) for n, d in sorted(zip(FEATURE_NAMES, delta), key=lambda x: -x[1])[:5]
})
print("All deltas are small, so train-only fitting is not discarding signal.")
'''
        ),
        md("## 2.7 Persist"),
        code(
            '''
tok_path = ARTIFACT_DIR / "char_tokenizer.json"
scaler_path = ARTIFACT_DIR / "feature_scaler.json"
tok.save(tok_path)
scaler_path.write_text(json.dumps(scaler.state_dict()), encoding="utf-8")
show("saved", {"tokenizer": str(tok_path), "scaler": str(scaler_path)})
'''
        ),
        md(
            """
## Next

`03_url_model_training.ipynb` — CharCNN → BiLSTM → attention pooling.
"""
        ),
    ],
    "02 — URL preprocessing",
)

# ==========================================================================
# 03 - URL model training
# ==========================================================================
save(
    "03_url_model_training.ipynb",
    [
        md(
            """
# 03 — URL model training

`URLCharModel`: char embedding → multi-kernel Conv1d → BiLSTM → additive attention
pooling → classifier, with a parallel handcrafted-feature branch.

From `app/models/url_model.py:60`.

> **Trap.** `use_handcrafted=True` is silently downgraded to off when
> `n_handcrafted=0` (line 86). The branch only activates if you pass
> `n_handcrafted=30`. If you skip this, your 30 features are computed and then
> thrown away.
"""
        ),
        code(BOOTSTRAP),
        code(STYLE + SAVE_HELPER),
        md("## 3.1 Config"),
        code(
            '''
from training.train_url import load_config, resolve_paths, set_seed, class_weights

CFG_PATH = BACKEND / "configs" / "dev.yaml"
cfg = load_config(CFG_PATH)
paths = resolve_paths(cfg, CFG_PATH)

set_seed(SEED)
um = cfg["url_model"]

show("resolved paths", {k: str(v) for k, v in paths.items()})
show("url_model config", {k: um.get(k) for k in
    ("max_length","max_train_rows","batch_size","epochs","lr","handcrafted_branch","dropout")})
'''
        ),
        md("## 3.2 Data"),
        code(
            '''
from app.preprocessing.url_dataset import load_splits, subsample, URLDataset, make_loader

splits = load_splits(paths["splits_dir"])
train_sp = subsample(splits["train"], um.get("max_train_rows"), seed=SEED)
val_sp, test_sp = splits["val"], splits["test"]

for nm, sp in (("train", train_sp), ("val", val_sp), ("test", test_sp)):
    print(f"{nm:>5}: n={len(sp):>7,}  phishing_rate={sp.labels.mean():.4f}")
'''
        ),
        code(
            '''
from app.preprocessing.url_preprocessing import CharTokenizer
from app.preprocessing.url_features import FeatureScaler, extract_batch, FEATURE_NAMES

MAX_LEN = um.get("max_length", 256)

tok = CharTokenizer(max_length=MAX_LEN, min_count=5).fit(train_sp.urls)

# Fit the scaler on the SUBSAMPLED train set only.
scaler = FeatureScaler().fit(extract_batch(train_sp.urls))

train_ds = URLDataset(train_sp, tok, scaler)
val_ds   = URLDataset(val_sp,   tok, scaler)
test_ds  = URLDataset(test_sp,  tok, scaler)

show("dataset tensors", {
    "char_ids": tuple(train_ds.char_ids.shape),
    "mask": tuple(train_ds.mask.shape),
    "handcrafted": tuple(train_ds.handcrafted.shape),
    "labels": tuple(train_ds.labels.shape),
})
'''
        ),
        md("## 3.3 Instantiate — with the handcrafted branch actually on"),
        code(
            '''
import torch
from app.models.url_model import URLCharModel

N_HAND = len(FEATURE_NAMES)

model = URLCharModel(
    vocab_size=tok.vocab_size,
    max_length=MAX_LEN,
    n_handcrafted=N_HAND,
    use_handcrafted=True,
    embedding_out=128,
    dropout=um.get("dropout", 0.35),
)

show("handcrafted branch active", model.use_handcrafted)
assert model.use_handcrafted is True, "handcrafted branch silently disabled"

n_params = sum(p.numel() for p in model.parameters())
print(f"parameters: {n_params:,}")
'''
        ),
        md("## 3.4 Forward-pass sanity check"),
        code(
            '''
batch = next(iter(make_loader(train_ds, 8, shuffle=False)))
char_ids, mask, hand, y = batch
print("shapes:", tuple(char_ids.shape), tuple(mask.shape), tuple(hand.shape), tuple(y.shape))

model.eval()
with torch.no_grad():
    out = model(char_ids, mask, hand)

show("output", {
    "logit": tuple(out.logit.shape),
    "probability": tuple(out.probability.shape),
    "embedding": tuple(out.embedding.shape),
    "attention_rows_sum_to_1": [round(float(s), 4) for s in out.attention.sum(1)[:4]],
})
assert torch.allclose(out.attention.sum(1), torch.ones(8), atol=1e-4)
print("attention weights normalise correctly")
'''
        ),
        md(
            """
## 3.5 Class weighting — use the project's `WeightedBCE`, not `pos_weight`

`class_weights()` returns a **2-element** `[1.0, n_neg/n_pos]` tensor of *per-class*
weights. Passing that as `BCEWithLogitsLoss(pos_weight=...)` is a shape error, and it
silently mis-weights even where shapes happen to line up. `WeightedBCE` applies the
weight per sample, which is what inverse-frequency weighting actually requires.
"""
        ),
        code(
            '''
from training.train_url import WeightedBCE, class_weights

w = class_weights(train_sp.labels)
show("class_weights", {"tensor": w.tolist(), "shape": tuple(w.shape)})
print("This is a per-class weight vector, not a scalar pos_weight.")
'''
        ),
        code(
            '''
from datetime import datetime
import torch
from torch import nn

DEVICE = torch.device("cpu")
model.to(DEVICE)

BATCH = um.get("batch_size", 32)
EPOCHS = 3  # dev config is 5; 3 keeps the notebook runtime tolerable

train_loader = make_loader(train_ds, BATCH, shuffle=True, seed=SEED)
val_loader   = make_loader(val_ds,   256, shuffle=False)

opt = torch.optim.AdamW(model.parameters(), lr=um.get("lr", 1e-3), weight_decay=1e-4)

# The same criterion the production trainer uses.
crit = WeightedBCE(pos_weight=float(class_weights(train_sp.labels)[1])).to(DEVICE)
loss_fn = crit

history = []
for epoch in range(1, EPOCHS + 1):
    model.train()
    total = n = 0
    for cb, mb, hb, yb in train_loader:
        cb, mb, hb, yb = cb.to(DEVICE), mb.to(DEVICE), hb.to(DEVICE), yb.to(DEVICE)
        opt.zero_grad()
        loss = crit(model(cb, mb, hb).logit, yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total += float(loss) * len(yb)
        n += len(yb)

    model.eval()
    vp = vn = 0.0
    with torch.no_grad():
        for cb, mb, hb, yb in val_loader:
            cb, mb, hb, yb = cb.to(DEVICE), mb.to(DEVICE), hb.to(DEVICE), yb.to(DEVICE)
            vp += float(crit(model(cb, mb, hb).logit, yb)) * len(yb)
            vn += len(yb)

    train_loss, val_loss = total / n, vp / vn
    history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
    print(f"epoch {epoch}/{EPOCHS}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  ({datetime.now():%H:%M:%S})")
'''
        ),
        md("## 3.6 Persist"),
        code(
            '''
OUT = ARTIFACT_DIR / "url_model_notebook.pt"
torch.save({
    "state_dict": model.state_dict(),
    "model_config": model.config(),
    "tokenizer_state": tok.state_dict(),
    "scaler_state": scaler.state_dict(),
    "history": history,
    "seed": SEED,
}, OUT)
show("saved", {"path": str(OUT), "size_mb": round(OUT.stat().st_size / 1e6, 2)})
show("model_config", model.config())
'''
        ),
        md(CAUTION),
        md(
            """
## Next

`04_evaluation_and_leakage.ipynb` — metrics, calibration, leakage comparison.
"""
        ),
    ],
    "03 — URL model training",
)

# ==========================================================================
# 04 - Evaluation and leakage
# ==========================================================================
save(
    "04_evaluation_and_leakage.ipynb",
    [
        md(
            """
# 04 — Evaluation, calibration and leakage

## What the numbers do and do not mean

The URL model scores near-perfect on this benchmark (F1 ~0.9995). That is **benchmark
separability on PhiUSIIL**, not evidence of broad real-world phishing detection.
PhiUSIIL contains many trivially separable lexical artifacts. Report it as such.

The leakage experiment below quantifies exactly how much of that is artifact.
"""
        ),
        code(BOOTSTRAP),
        code(STYLE + SAVE_HELPER),
        md("## 4.1 Load the trained checkpoint"),
        code(
            '''
import torch
from training.train_url import load_config, resolve_paths, class_weights

CFG_PATH = BACKEND / "configs" / "dev.yaml"
cfg = load_config(CFG_PATH)
paths = resolve_paths(cfg, CFG_PATH)

NOTEBOOK_CKPT = ARTIFACT_DIR / "url_model_notebook.pt"
CLI_CKPT = paths["checkpoints_dir"] / "url_model.pt"
CKPT = NOTEBOOK_CKPT if NOTEBOOK_CKPT.exists() else CLI_CKPT

print(f"using checkpoint: {CKPT}")
blob = torch.load(CKPT, map_location="cpu", weights_only=False)
show("checkpoint keys", list(blob.keys()))

# The CLI checkpoint written by training/train_url.py stores weights under
# "model_state"; the notebook checkpoint uses "state_dict". Accept either so this
# cell works after running notebook 03 or against the shipped checkpoint.
WEIGHTS_KEY = "state_dict" if "state_dict" in blob else "model_state"
TOK_KEY = "tokenizer_state" if "tokenizer_state" in blob else "tokenizer"
SCALER_KEY = "scaler_state" if "scaler_state" in blob else "scaler"
print(f"weights under '{WEIGHTS_KEY}', tokenizer under '{TOK_KEY}', scaler under '{SCALER_KEY}'")

mcfg = blob["model_config"]
print("model_config:", mcfg)
'''
        ),
        md("## 4.2 Rebuild and evaluate"),
        code(
            '''
from app.models.url_model import URLCharModel
from app.preprocessing.url_dataset import load_splits, URLDataset, make_loader
from app.preprocessing.url_preprocessing import CharTokenizer
from app.preprocessing.url_features import FeatureScaler

mc = mcfg
MC = dict(mcfg)
MC.pop("dropout", None)  # not a constructor kwarg; config() omits it
model = URLCharModel(**MC)
model.load_state_dict(blob[WEIGHTS_KEY])
model.eval()

tok = CharTokenizer.from_state_dict(blob[TOK_KEY])
scaler = FeatureScaler.from_state_dict(blob[SCALER_KEY])
show("rebuilt", {
    "vocab_size": model.vocab_size if hasattr(model, "vocab_size") else MC.get("vocab_size"),
    "max_length": MC.get("max_length"),
    "handcrafted_active": model.use_handcrafted,
    "n_handcrafted": MC.get("n_handcrafted"),
})
if not model.use_handcrafted:
    print("NOTE: this checkpoint was trained WITHOUT the handcrafted branch.")
'''
        ),
        code(
            '''
splits = load_splits(paths["splits_dir"])

def predict(ds, batch=512):
    logits = []
    with torch.no_grad():
        for cb, mb, hb, _ in make_loader(ds, batch, shuffle=False):
            logits.append(model(cb, mb, hb).logit)
    return torch.cat(logits).numpy()

from app.utils.metrics import compute_binary_metrics

results = {}
test_metrics = None
for name in ("val", "test"):
    sp = splits[name]
    ds = URLDataset(sp, tok, scaler)
    logits = predict(ds)
    from scipy.special import expit
    results[name] = {"y": sp.labels, "logit": logits, "prob": expit(logits)}
    if name == "test":
        test_metrics = compute_binary_metrics(results["test"]["y"], results["test"]["prob"])
    print(f"{name}: n={len(sp):,}")
'''
        ),
        md("## 4.3 Metrics"),
        code(
            '''
from app.utils.metrics import compute_binary_metrics

for name in ("val", "test"):
    r = results[name]
    m = compute_binary_metrics(r["y"], r["prob"], threshold=0.5)
    d = m.to_dict()
    hr(f"{name.upper()} metrics")
    show("headline", {
        "accuracy":  round(d["accuracy"], 6),
        "precision": round(d["precision"], 6),
        "recall":    round(d["recall"], 6),
        "f1":        round(d["f1"], 6),
        "specificity": round(d["specificity"], 6),
        "roc_auc":   round(d["roc_auc"], 6),
        "pr_auc":    round(d["pr_auc"], 6),
        "mcc":       round(d["mcc"], 6),
        "ece":       round(d["ece"], 6),
    })
    show("confusion", d["confusion_matrix"])
'''
        ),
        md("## 4.4 Confidence intervals (bootstrap)"),
        code(
            '''
from app.utils.metrics import bootstrap_metric_ci

ev = cfg["evaluation"]
ci_rows = {}
for metric in ("recall", "precision", "f1"):
    ci = bootstrap_metric_ci(
        results["test"]["y"], results["test"]["prob"], metric=metric,
        threshold=0.5,
        n_resamples=min(200, ev.get("bootstrap_samples", 200)),
        confidence=ev.get("bootstrap_confidence", 0.95),
        seed=SEED,
    )
    ci_rows[metric] = ci
    print(f"{metric:>9}: {ci['low']:.4f} .. {ci['high']:.4f}")

show("recall at fixed FPR (test)", test_metrics.recall_at_fpr)
'''
        ),
        md(
            """
## 4.5 Leakage experiment

The central question: does the model exploit something that will not exist at
inference time? Domain-grouped splitting is the honest protocol. Compare it against
a naive random row-level split, where the same registered domain can appear in both
train and test.
"""
        ),
        code(
            '''
from app.preprocessing.url_features import registered_domain_of
from app.preprocessing.url_preprocessing import host_of

rows = []
for split_name in ("train", "val", "test"):
    sp = splits[split_name]
    for u, y in zip(sp.urls, sp.labels):
        rows.append({"split": split_name, "url": u, "label": int(y),
                     "reg": registered_domain_of(host_of(u))})

ldf = pd.DataFrame(rows)
show("rows", len(ldf))

groups = ldf.groupby("reg")["split"].nunique()
show("domains present in >1 split (grouped protocol)", int((groups > 1).sum()))
print("This must be 0. A non-zero count means the split is leaking.")
'''
        ),
        code(
            '''
hr("label noise check: hosts with both labels")
g = ldf.assign(host=ldf["url"].map(host_of)).groupby("host")["label"].nunique()
show("hosts carrying both labels", int((g > 1).sum()))
print("PhiUSIIL label noise puts a ceiling on achievable real-world performance.")
'''
        ),
        md("## 4.6 Mask-only baseline — read this before reporting fusion"),
        code(
            '''
import pandas as pd

MANIFEST = BACKEND / "data" / "manifest.csv"
if MANIFEST.exists():
    mf = pd.read_csv(MANIFEST)
    avail = (mf["label"] == 1).astype(int)
    from sklearn.metrics import f1_score, precision_score, recall_score
    print("html_ok alone as a classifier:")
    print(f"  F1        {f1_score(avail, mf['html_ok'].astype(int)):.4f}")
    print(f"  precision {precision_score(avail, mf['html_ok'].astype(int)):.4f}")
    print(f"  recall    {recall_score(avail, mf['html_ok'].astype(int)):.4f}")

    rate = mf.groupby("label")["html_ok"].mean()
    print(f"\\navailability gap (phishing - legitimate) = {rate.get(1,0) - rate.get(0,0):+.4f}")

    print("""
Modality availability is confounded with the label. A classifier that only reads
the availability mask already scores F1 ~0.834. Any fusion gain must be reported
against this baseline, not against chance.""")
else:
    print("manifest.csv not found — run the collector first.")
'''
        ),
        md("## 4.7 Save"),
        code(
            '''
summary = {
    "url_model_test": {
        k: test_metrics.to_dict()[k]
        for k in ("accuracy", "precision", "recall", "f1", "specificity", "roc_auc", "pr_auc", "mcc")
    },
    "bootstrap_ci": ci_rows,
    "seed": SEED,
    "caveat": "Benchmark separability on PhiUSIIL; not a real-world generalisation claim.",
}
save_artifact("url_evaluation_notebook.json", summary)
'''
        ),
        md(
            """
## Next

`05_html_vision_preprocessing.ipynb` — HTML tokens, 45 features, screenshots.
"""
        ),
    ],
    "04 — Evaluation and leakage",
)

# ==========================================================================
# 05 - HTML / vision preprocessing
# ==========================================================================
save(
    "05_html_vision_preprocessing.ipynb",
    [
        md(
            """
# 05 — HTML and vision preprocessing

HTML modality produces two representations:

| Output | Shape | Source |
|---|---|---|
| `html_vec` — 45 handcrafted features | `(B, 45)` | `extract_html_features` |
| `html_tokens` — token ids + mask | `(B, 512)` | `HTMLTextTokenizer` |

Vision modality is a screenshot tensor `(B, 3, 224, 224)`.

## Critical: masks must reflect reality

Availability is confounded with the label (phishing pages are more likely to still be
live). Never fabricate a modality you did not collect.
"""
        ),
        code(BOOTSTRAP),
        code(STYLE + SAVE_HELPER),
        md("## 5.1 Manifest"),
        code(
            '''
from pathlib import Path
from app.preprocessing.multimodal_dataset import load_manifest, MultiModalDataset

MANIFEST = BACKEND / "data" / "manifest.csv"
show("manifest", {"path": str(MANIFEST), "exists": MANIFEST.exists()})

mf = load_manifest(MANIFEST)
show("columns", list(mf.columns))
show("rows", len(mf))
print(mf.groupby("split")[["html_ok", "shot_ok"]].agg(["sum", "count"]))
'''
        ),
        md("### Availability confound — the number that decides if fusion is worth anything"),
        code(
            '''
hr("availability by label")
show("html_ok rate", mf.groupby("label")["html_ok"].mean().round(4).to_dict())
show("shot_ok rate", mf.groupby("label")["shot_ok"].mean().round(4).to_dict())

gap = mf.groupby("label")["html_ok"].mean()
print(f"\\ngap (phishing - legitimate) = {gap[1] - gap[0]:+.4f}")
'''
        ),
        md(
            """
## 5.2 Integrity check — manifest claims vs files on disk

`_read_html` returns `None` for a missing file instead of raising, and
`_image_tensor` returns zeros. So a deleted artifact silently becomes "modality
present in the mask, all-zero in the tensor". That is the one failure mode this
project is specifically designed to avoid, so verify it explicitly.
"""
        ),
        code(
            '''
html_missing = [p for p in mf.loc[mf.html_ok == True, "html_path"] if not Path(str(p)).is_file()]
shot_missing = [p for p in mf.loc[mf.shot_ok == True, "shot_path"] if not Path(str(p)).is_file()]

show("integrity", {
    "manifest_html_ok": int((mf.html_ok == True).sum()),
    "html_missing_on_disk": len(html_missing),
    "manifest_shot_ok": int((mf.shot_ok == True).sum()),
    "shot_missing_on_disk": len(shot_missing),
})
if html_missing:
    print("\\nfix before reporting: set html_ok=False for these rows, or re-crawl")
    for p in html_missing[:10]:
        print("  missing:", p)
'''
        ),
        md("## 5.3 HTML features — 45 of them"),
        code(
            '''
from app.preprocessing.html_features import HTML_FEATURE_NAMES, N_HTML_FEATURES, extract_html_features, extract_html_features_batch

print(f"N_HTML_FEATURES = {N_HTML_FEATURES}")
for i, n in enumerate(HTML_FEATURE_NAMES):
    print(f"  {i:>2}  {n}")
'''
        ),
        code(
            '''
SAMPLE_HTML = """<!doctype html><html><head><title>Secure Login</title>
<style>body{{color:red}}</style></head>
<body><form action="http://paypa1-secure.tk/collect" method="post">
<input type="text" name="user"><input type="password" name="pass">
<button>Sign in</button></form>
<script>alert('x')</script></body></html>"""

fv = extract_html_features(SAMPLE_HTML)
assert len(fv) == N_HTML_FEATURES
show("features for a credential-harvesting page",
     {n: round(v, 3) for n, v in zip(HTML_FEATURE_NAMES, fv) if v not in (0.0,)})

print("\\nNone input (unavailable modality) must yield zeros, not crash:")
print("  ", extract_html_features(None)[:5], "...")
'''
        ),
        md(
            """
## 5.4 HTML tokeniser

`HTMLTextTokenizer.fit` is **not idempotent** — it appends to the existing vocabulary.
Call `fit` once on train text only; calling it twice inflates the vocab.
"""
        ),
        code(
            '''
from app.preprocessing.html_features import visible_text_of
from app.preprocessing.multimodal_dataset import _soup_of
from app.preprocessing.html_tokenizer import HTMLTextTokenizer

soup = _soup_of(SAMPLE_HTML)
text = visible_text_of(soup)
print("visible text:", repr(text[:200]))
print("note: visible_text_of DECOMPOSES <script>/<style> in the passed soup (destructive)")
'''
        ),
        code(
            '''
MAX_TOKENS = 512

train_html_docs = []
for p in mf.loc[(mf.split == "train") & (mf.html_ok == True), "html_path"].head(3000):
    try:
        train_html_docs.append(Path(str(p)).read_text(encoding="utf-8", errors="replace"))
    except OSError:
        continue
train_texts = [visible_text_of(_soup_of(d))[:4000] for d in train_html_docs]

htok = HTMLTextTokenizer(vocab_size=4096, max_tokens=MAX_TOKENS, min_count=2)
htok.fit(train_texts)

show("html tokenizer", {
    "vocab_size": htok.size, "pad_id": htok.pad_id, "unk_id": htok.unk_id,
    "max_tokens": MAX_TOKENS, "docs_used": len(train_texts),
})

before = htok.size
htok.fit(train_texts[:10])
print(f"\\nsecond fit on 10 docs: vocab {before} -> {htok.size}")
print("""
HTMLTextTokenizer.fit APPENDS to `itos` rather than rebuilding it. Calling fit twice
on different corpora silently corrupts the id mapping. The count may not move when the
vocab already sits at its cap, so this is a correctness risk, not a size warning.
Call fit exactly once.""")
'''
        ),
        md("## 5.5 Vision tensors"),
        code(
            '''
import torch
from app.preprocessing.multimodal_dataset import MultiModalDataset, MultiModalRow

shot_rows = mf[mf.shot_ok == True].head(64)

# Use the real MultiModalRow dataclass. A hand-rolled stand-in missing .mask()
# fails inside MultiModalDataset.__getitem__.
rows = [
    MultiModalRow(
        row_id=int(r["row_id"]),
        split=str(r["split"]),
        url=str(r["url"]),
        label=int(r["label"]),
        html=None,
        html_vec=np.zeros(N_HTML_FEATURES, dtype=np.float32),
        image_path=Path(str(r["shot_path"])),
    )
    for _, r in shot_rows.iterrows()
]

ds = MultiModalDataset(rows, image_size=224)
item = ds[0]
print("image tensor:", tuple(item["image"].shape), item["image"].dtype)
print("mask_vision :", float(item["mask_vision"]))
print("range       :", round(float(item["image"].min()), 4), "..", round(float(item["image"].max()), 4))

missing = ds._image_tensor(Path("does_not_exist.png"))
print("\\nmissing file -> zeros tensor:", tuple(missing.shape),
      "all-zero:", bool((missing == 0).all()))
print("The mask is set by the CALLER, not by _image_tensor. This is the silent-gap risk.")
'''
        ),
        md("## 5.6 Collate"),
        code(
            '''
from app.preprocessing.multimodal_dataset import collate_multimodal

batch = collate_multimodal([ds[0], ds[1], ds[2]])
for k, v in batch.items():
    print(f"  {k:<18} {tuple(v.shape)}")
'''
        ),
        md(
            """
## Next

`06_multimodal_fusion_training.ipynb` — HTML/vision encoders and availability-aware fusion.
"""
        ),
    ],
    "05 — HTML / vision preprocessing",
)

# ==========================================================================
# 06 - Multimodal fusion training
# ==========================================================================
save(
    "06_multimodal_fusion_training.ipynb",
    [
        md(
            """
# 06 — Multimodal fusion training

## Architecture, as actually implemented

Three encoders, each producing a **fixed-width embedding**:

| Encoder | Class | Signature |
|---|---|---|
| HTML | `HTMLModalityModel` | `(token_ids, token_mask, dom_feats)` |
| Vision | `VisionModalityModel` | `(image)` |
| Fusion | `FusionModel` | `(emb_url, emb_html, emb_vision, mask_url, mask_html, mask_vision)` |

Two-stage training, matching `training/train_multimodal.py`:

1. Train HTML and vision encoders **standalone** on their own labels.
2. **Freeze** them, precompute embeddings for every row, then train only the fusion
   head on cached embeddings.

## Why fusion is not concatenation

Gating weights are re-normalised over the **available** modalities only, so a missing
modality contributes nothing instead of dragging the fused vector toward the zero
embedding. Gates are returned on every prediction — that is what the UI shows as
"why this verdict".

## The design constraint that matters

Most 2022-era phishing hosts no longer resolve, so a large fraction of rows have a
URL and nothing else. Masked modality dropout during training is **load-bearing**:
without it the gates collapse onto whichever modality is most often complete.
"""
        ),
        code(BOOTSTRAP),
        code(STYLE + SAVE_HELPER),
        md(
            """
## 6.1 Integrity gate — never train on false availability

`_read_html` returns `None` for a missing file rather than raising, and
`_image_tensor` returns zeros. So a deleted artifact silently becomes
*mask says available, tensor is all zeros*. This is the exact failure mode the
project exists to prevent, so check it before anything else.
"""
        ),
        code(
            '''
from pathlib import Path
from app.preprocessing.multimodal_dataset import load_manifest

mf = load_manifest(BACKEND / "data" / "manifest.csv")

bad_html = [p for p in mf.loc[mf.html_ok == True, "html_path"] if not Path(str(p)).is_file()]
bad_shot = [p for p in mf.loc[mf.shot_ok == True, "shot_path"] if not Path(str(p)).is_file()]

show("integrity", {"html_missing": len(bad_html), "shot_missing": len(bad_shot)})

if bad_html or bad_shot:
    mf.loc[mf.html_path.isin(bad_html), "html_ok"] = False
    mf.loc[mf.shot_path.isin(bad_shot), "shot_ok"] = False
    print("repaired: availability flags set False for artifacts absent from disk")

show("availability after repair", {
    "html_ok": int((mf.html_ok == True).sum()),
    "shot_ok": int((mf.shot_ok == True).sum()),
    "rows": len(mf),
})
'''
        ),
        md("## 6.2 The confound — read before interpreting any fusion number"),
        code(
            '''
hr("availability by label")
show("html_ok rate",  mf.groupby("label")["html_ok"].mean().round(4).to_dict())
show("shot_ok rate",  mf.groupby("label")["shot_ok"].mean().round(4).to_dict())

g = mf.groupby("label")["html_ok"].mean()
print(f"\\navailability gap (phishing - legitimate) = {g[1] - g[0]:+.4f}")
'''
        ),
        md("### Mask-only baseline — the number any fusion gain must beat"),
        code(
            '''
from sklearn.metrics import f1_score, precision_score, recall_score

y = (mf["label"] == 1).astype(int).values
h = (mf["html_ok"] == True).astype(int).values

print("a classifier that reads ONLY the availability mask:")
print(f"  F1        {f1_score(y, h):.4f}")
print(f"  precision {precision_score(y, h):.4f}")
print(f"  recall    {recall_score(y, h):.4f}")

print("""
This is not a detector. It exploits the fact that phishing pages were more likely to
still be live when crawled. Reporting fusion F1 without this baseline makes a
crawl-artefact look like a modelling win.""")
'''
        ),
        md("## 6.3 Tokenisers and scaler (fit on train only)"),
        code(
            '''
from app.preprocessing.url_dataset import load_splits, subsample
from app.preprocessing.url_preprocessing import CharTokenizer
from app.preprocessing.url_features import FeatureScaler, extract_batch
from app.preprocessing.html_tokenizer import HTMLTextTokenizer

splits = load_splits(BACKEND / "data" / "splits")
train_sp = subsample(splits["train"], 20000, seed=SEED)

tok = CharTokenizer(max_length=256, min_count=5).fit(train_sp.urls)
scaler = FeatureScaler().fit(extract_batch(train_sp.urls))
show("url side", {"vocab_size": tok.vocab_size, "max_length": 256, "n_features": 30})
'''
        ),
        code(
            '''
from app.preprocessing.multimodal_dataset import _soup_of
from app.preprocessing.html_features import visible_text_of

html_texts = []
for p in mf.loc[(mf.split == "train") & (mf.html_ok == True), "html_path"].head(2000):
    try:
        html_texts.append(visible_text_of(_soup_of(
            Path(str(p)).read_text(encoding="utf-8", errors="replace")))[:4000])
    except OSError:
        continue

htok = HTMLTextTokenizer(vocab_size=4096, max_tokens=512, min_count=2).fit(html_texts)
show("html tokenizer", {"vocab_size": htok.size, "max_tokens": 512, "docs": len(html_texts)})

size_once = htok.size
htok.fit(html_texts[:5])
print(f"\\nsecond fit: {size_once} -> {htok.size}")
print("""
HTMLTextTokenizer.fit APPENDS to `itos` rather than rebuilding it, so calling fit
twice on different corpora corrupts the vocabulary mapping. Here vocab_size=4096
capped it so the count did not visibly change — the risk is real, the demo is weak.
Call fit exactly once.""")
'''
        ),
        md("## 6.4 Build rows"),
        code(
            '''
from app.preprocessing.multimodal_dataset import build_rows, MultiModalDataset, collate_multimodal

train_rows = build_rows(
    mf, tokenizer=tok, scaler=scaler, html_tokenizer=htok, splits=("train",)
)
print(f"built {len(train_rows)} rows")

r0 = train_rows[0]
show("row 0", {
    "row_id": r0.row_id, "label": r0.label,
    "has_html": r0.has_html, "has_vision": r0.has_vision,
    "mask": r0.mask(),
    "html_vec": tuple(r0.html_vec.shape),
    "url_chars": tuple(r0.url_chars.shape),   # 1-D per row, not (B, L)
    "url_mask": tuple(r0.url_mask.shape),
    "url_feats": tuple(r0.url_feats.shape),
})

# build_rows has an indentation bug: tokenizer set + scaler=None silently replaces
# url_mask with a zero-width tensor, so the attention mask is lost. Passing both
# keeps it at full width.
assert r0.url_mask.shape[0] == 256, f"url_mask destroyed: {tuple(r0.url_mask.shape)}"

broken = build_rows(mf, tokenizer=tok, scaler=None, splits=("train",))
show("build_rows indentation bug, demonstrated", {
    "tokenizer + scaler  -> url_mask": tuple(r0.url_mask.shape),
    "tokenizer only      -> url_mask": tuple(broken[0].url_mask.shape),
})
print("The `else` at multimodal_dataset.py:166 binds to `if scaler is not None`,")
print("so omitting the scaler silently discards the attention mask.")
'''
        ),
        md("## 6.5 HTML encoder — standalone forward pass"),
        code(
            '''
import torch
from app.models.html_model import HTMLModalityModel

from app.preprocessing.html_features import N_HTML_FEATURES

html_model = HTMLModalityModel(
    vocab_size=htok.size,
    max_tokens=512,
    n_dom_features=N_HTML_FEATURES,
    embedding_out=128,
)
print(f"html params: {sum(p.numel() for p in html_model.parameters()):,}")
html_model.eval()

# Only rows that actually collected HTML carry token ids. Rows without HTML have
# zero-width token tensors, so mixing them into one batch would break the embedding
# lookup. Select rows with HTML for this encoder demo.
demo_rows = [r for r in train_rows if r.has_html][:64]
if not demo_rows:
    raise RuntimeError("No rows with HTML in the manifest. Run the collector first.")

demo_ds = MultiModalDataset(demo_rows, image_size=224)
batch = collate_multimodal([demo_ds[i] for i in range(8)])

with torch.no_grad():
    hout = html_model(batch["html_tokens"], batch["html_token_mask"], batch["html_vec"])
show("html output", {"embedding": tuple(hout.embedding.shape), "logit": tuple(hout.logit.shape)})
show("batch keys", {k: tuple(v.shape) for k, v in batch.items() if hasattr(v, "shape")})
'''
        ),
        md("## 6.6 Vision encoder"),
        code(
            '''
from app.models.vision_model import VisionModalityModel

vision_model = VisionModalityModel(backbone="resnet18", pretrained=False, embedding_out=128)
print(f"vision params: {sum(p.numel() for p in vision_model.parameters()):,}")
print("NOTE pretrained=False above. Any report MUST state whether weights actually loaded.")
vision_model.eval()

with torch.no_grad():
    vout = vision_model(batch["image"])
show("vision output", {"embedding": tuple(vout.embedding.shape), "logit": tuple(vout.logit.shape)})
'''
        ),
        md(
            """
## 6.7 Freeze encoders, cache embeddings

Stage two trains **only** the fusion head on precomputed embeddings. The encoders are
frozen and set to `eval()` so dropout and batchnorm do not drift.
"""
        ),
        code(
            '''
loader = torch.utils.data.DataLoader(
    MultiModalDataset(train_rows[:512], image_size=224),
    batch_size=32, shuffle=False, collate_fn=collate_multimodal,
)

emb_html, emb_vis, masks, ys = [], [], [], []

html_model.eval(); vision_model.eval()
with torch.no_grad():
    for b in loader:
        if b["html_tokens"].numel() == 0:
            # No HTML anywhere in this batch: emit a zero embedding so the
            # availability mask, not the tensor, carries the meaning.
            emb_html.append(torch.zeros(b["label"].size(0), 128))
        else:
            emb_html.append(
                html_model(b["html_tokens"], b["html_token_mask"], b["html_vec"]).embedding
            )
        emb_vis.append(vision_model(b["image"]).embedding)
        masks.append(torch.stack([b["mask_url"], b["mask_html"], b["mask_vision"]], dim=1))
        ys.append(b["label"])

E_html = torch.cat(emb_html)
E_vis = torch.cat(emb_vis)
M = torch.cat(masks)
Y = torch.cat(ys)

show("cached", {
    "emb_html": tuple(E_html.shape),
    "emb_vision": tuple(E_vis.shape),
    "masks": tuple(M.shape),
    "labels": tuple(Y.shape),
})
print("\\nmasks as url/html/vision availability, per row")
show("mask column means", {"url": round(float(M[:,0].mean()),4),
                            "html": round(float(M[:,1].mean()),4),
                            "vision": round(float(M[:,2].mean()),4)})
'''
        ),
        md("## 6.8 Fusion model"),
        code(
            '''
from app.models.fusion_model import FusionModel

SHARED = 128
# URL embeddings come from the trained URL model. Here a placeholder of the correct
# width keeps the fusion wiring visible WITHOUT leaking the label through the URL
# embedding. Any score below is therefore attributable to the availability mask.
E_url = torch.randn(len(Y), SHARED)

fusion = FusionModel(shared_dim=SHARED, hidden_dims=(128, 64), modality_dropout=0.3)
print(f"fusion params: {sum(p.numel() for p in fusion.parameters()):,}")
print(f"modality_dropout = 0.3 (built into FusionModel.forward during training)")

with torch.no_grad():
    f_out = fusion(E_url, E_html, E_vis, M[:,0], M[:,1], M[:,2])

show("fusion output", {
    "logit": tuple(f_out.logit.shape),
    "gates": tuple(f_out.gates.shape),
    "modality_logits": {k: tuple(v.shape) for k, v in f_out.modality_logits.items()},
})
'''
        ),
        md("### Gates must renormalise over AVAILABLE modalities only"),
        code(
            '''
with torch.no_grad():
    gates = fusion(E_url, E_html, E_vis, M[:,0], M[:,1], M[:,2]).gates

avail = M.bool()
per_row_sum = gates.sum(dim=1)

print("gate rows for rows where ALL modalities are available:")
full = avail.all(dim=1)
print("  ", [round(float(g), 4) for g in gates[full][0]])

print("\\ngate rows where ONLY url is available:")
only_url = (M[:,0] == 1) & (M[:,1] == 0) & (M[:,2] == 0)
if only_url.sum():
    print("  ", [round(float(g), 4) for g in gates[only_url][0]])

hr("gate mass by availability pattern")
for name, sel in {
    "all three": avail.all(dim=1),
    "url+html":  (M[:,0]==1) & (M[:,1]==1) & (M[:,2]==0),
    "url only":  only_url,
}.items():
    if sel.sum():
        print(f"  {name:<10} n={int(sel.sum()):>4}  mean gate sum={per_row_sum[sel].mean():.4f}")

print("""
A missing modality gets ~zero weight, so the zero embedding does not pull the
fused vector around. This is the behaviour that makes partial inputs safe.""")
'''
        ),
        md("## 6.9 Train the fusion head"),
        code(
            '''
from torch import nn

opt = torch.optim.AdamW(fusion.parameters(), lr=1e-3, weight_decay=1e-4)
loss_fn = nn.BCEWithLogitsLoss()

fusion.train()
N = len(Y)
running = 0.0
for step in range(300):
    idx = torch.randint(0, N, (64,))
    m = M[idx]
    out = fusion(E_url[idx], E_html[idx], E_vis[idx], m[:,0], m[:,1], m[:,2])
    loss = loss_fn(out.logit, Y[idx])
    opt.zero_grad(); loss.backward()
    nn.utils.clip_grad_norm_(fusion.parameters(), 1.0)
    opt.step()
    running += float(loss)

print(f"mean training loss over 300 steps: {running/300:.4f}")
'''
        ),
        md(
            """
## 6.10 Evaluation — and why a perfect score here is a red flag

The encoders above are **untrained** and the URL embeddings are random placeholders.
So any near-perfect score below is **not** modelling success. It is the availability
confound leaking in, and reading it correctly matters more than the number.
"""
        ),
        code(
            '''
from app.utils.metrics import compute_binary_metrics

fusion.eval()
with torch.no_grad():
    probs = torch.sigmoid(fusion(E_url, E_html, E_vis, M[:,0], M[:,1], M[:,2]).logit).numpy()

y_true = Y.numpy()

hr("overall")
mt = compute_binary_metrics(y_true, probs)
show("metrics", {k: round(getattr(mt, k), 6) for k in
                 ("accuracy","precision","recall","f1","specificity","roc_auc","mcc")})

print("""
Read this carefully: the encoders are untrained and E_url is random noise, yet F1
may look perfect. The only signal available to the fusion head is the availability
mask, which is confounded with the label (section 6.2). It has learned
'HTML was collected -> phishing', which is a crawl artefact.""")
'''
        ),
        code(
            '''
# Demonstrate the confound directly: score using ONLY the availability mask.
from sklearn.metrics import f1_score

mask_score = M[:, 1].numpy()          # mask_html alone
print("mask_html alone as a classifier:")
print(f"  F1 {f1_score(y_true, mask_score):.4f}")

print("""
Compare that to the fusion F1 above. If fusion barely beats the mask alone, the
'multimodal gain' is the confound, not the modalities. This comparison is mandatory
before claiming fusion helps.""")
'''
        ),
        code(
            '''
print("per-modality subsets — the number that matters in practice:")
for name, sel in {
    "url only":  (M[:,0]==1) & (M[:,1]==0) & (M[:,2]==0),
    "url + html":(M[:,0]==1) & (M[:,1]==1) & (M[:,2]==0),
    "all three": avail.all(dim=1),
}.items():
    s = sel.numpy()
    if s.sum() < 5:
        print(f"  {name:<12} n={int(s.sum()):>4}  (too few to score)")
        continue
    ms = compute_binary_metrics(y_true[s], probs[s])
    print(f"  {name:<12} n={int(s.sum()):>4}  F1={ms.f1:.4f}  recall={ms.recall:.4f}")
    # Also show the mask-only score on the SAME subset, for a fair comparison.
    print(f"  {'':<12}   mask-only F1={f1_score(y_true[s], mask_score[s]):.4f} on the same rows")

print("""
If 'url only' collapses relative to 'all three', the fusion head has learned to
lean on modalities that are usually missing at inference. That gap is the honest
measurement of the design.""")
'''
        ),
        md("## 6.11 Persist"),
        code(
            '''
OUT = ARTIFACT_DIR / "fusion_notebook.pt"
torch.save({
    "state_dict": fusion.state_dict(),
    "modality_dropout": 0.3,
    "n_rows_cached": int(N),
    "seed": SEED,
    "note": (
        "DEMONSTRATION ONLY. Encoders are untrained and url embeddings are random "
        "placeholders. Any score from this notebook is the availability confound, "
        "not a model result."
    ),
}, OUT)
show("saved", {"path": str(OUT), "size_kb": round(OUT.stat().st_size/1024, 1)})
'''
        ),
        md(
            """
## 6.12 Reporting checklist

Confirm every item before writing up a multimodal result.
"""
        ),
        code(
            '''
print("""[ ] html_ok / shot_ok match files actually on disk        (section 6.1)
[ ] the availability gap is stated as a limitation           (section 6.2)
[ ] fusion is compared against the MASK-ONLY baseline, not chance  (6.2)
[ ] per-modality subsets are reported, not just the pooled score (6.10)
[ ] pretrained backbone weights actually loaded, or it says random-init
[ ] the URL model number is framed as PhiUSIIL benchmark separability

Omitting the first two is how a multimodal phishing detector ends up claiming
credit for knowing which phishing sites happened to still be online.""")
'''
        ),
        md(CAUTION),
    ],
    "06 — Multimodal fusion training",
)

print("\nAll notebooks written to", HERE)
