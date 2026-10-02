# Multimodal Phishing URL Detector

A research-grade phishing URL detection system. A submitted URL is analysed
through three independent modalities — the **raw URL string**, the **live HTML**,
and a **rendered screenshot** — fused by a mask-aware adaptive gating network
into a calibrated probability, with explainability derived from the real models
rather than from templated text.

> **Status: all phases implemented and run. Read the caveats.** The URL model,
> the HTML/vision/fusion models, the API, and the frontend are all built and
> trained on real data. Every number in `backend/reports/` comes from a real run.
>
> **The headline finding is a negative one, and it matters more than the scores:**
> on this dataset the URL string alone saturates the task, page content adds
> nothing measurable, and the multimodal result is inflated by a confound where
> whether a page could be collected largely *is* the label. See
> **Multimodal results (measured)** below and `backend/docs/limitations.md`.

---

## The dataset (measured)

`Dataset/Phishing_URL_Dataset.csv` — PhiUSIIL.

| Field | Value |
| --- | --- |
| Size / SHA-256 | 54,160,642 bytes / `748ae7c7…d79438ba` |
| Rows / columns | 235,795 / 55 |
| URL column (detected) | `URL` |
| Label column (detected) | `label` → `1` phishing, `0` legitimate |
| Class balance | 134,850 phishing (57.19 %) / 100,945 legitimate (42.81 %) |
| Duplicate full rows removed | 425 |
| Registered domains | 175,510 (120 carry both labels) |

Split by **registered domain**, seed 42, fingerprint `514839cb599158da`:

| Split | Rows | Domains | Phishing share |
| --- | ---: | ---: | ---: |
| train | 164,760 | 114,190 | 0.572930 |
| val | 35,305 | 29,760 | 0.572922 |
| test | 35,305 | 31,560 | 0.572922 |

Verified: **zero** domain overlap between any two splits. Full detail in
`backend/reports/dataset_inspection.md`.

## The URL model (measured)

A character-level CharCNN + BiLSTM with additive attention pooling over the
normalised URL, plus a 30-feature handcrafted branch computed from the URL string
alone. Trained on the **40,000-row dev subset** (of 164,760 available) because
this host has no usable GPU; the architecture is identical to the `full` profile.

Test split, Platt calibration fitted on validation only:

| Metric | Value | 95 % bootstrap CI |
| --- | ---: | --- |
| Phishing recall | 0.99990 | 0.99980 – 1.00000 |
| Precision | 0.99916 | 0.99877 – 0.99956 |
| F1 | 0.99953 | 0.99935 – 0.99973 |
| Specificity | 0.99887 | — |
| ROC-AUC | 0.99983 | — |
| PR-AUC | 0.99978 | — |
| MCC | 0.99890 | — |
| Brier | 0.00053 | — |
| ECE | 0.00020 | — |

Recall is 1.00 at a 1 % false-positive rate. Calibration reduced ECE from
0.00263 (uncalibrated) to 0.00020; Platt was selected over temperature and
isotonic on validation Brier.

**These numbers are unusually high, and that is a property of PhiUSIIL, not
evidence of a strong model.** The dataset is known to be largely separable from
the URL string alone, which is exactly why the leakage experiment below was run.

## Leakage experiment (measured)

Trains on PhiUSIIL's 31 page-derived engineered columns — information produced by
*fetching the page*, which a URL-only detector never has — and compares against
the shipped model on the identical test split.

| Model | Phishing recall | ROC-AUC |
| --- | ---: | ---: |
| URL-only (shipped) | 0.99990 | 0.99983 |
| Engineered-column MLP | 0.99975 | 0.99999 |
| **Delta** | **−0.00015** | — |

The engineered model is **not better** despite having strictly more information.
The leak is real in the data but produces no headline gain, so the reported
figure is not inflated by it. Full detail, including the single columns that
nearly separate the classes on their own, in
`backend/reports/leakage_experiment.md`.

## Multimodal results (measured, and mostly negative)

2,100 URLs were collected (HTML, then rendered screenshot). Of those, 1,001
returned HTML (47.7 %) and 970 yielded both modalities. The ResNet-18 backbone
did load pretrained weights (`pretrained_loaded: true`).

Test set, 700 rows:

| model | subset | n | F1 | specificity | ROC-AUC |
| --- | --- | ---: | ---: | ---: | ---: |
| URL | unrestricted | 700 | 1.000 | 1.000 | 1.000 |
| HTML | unrestricted | 700 | 0.724 | 0.030 | 0.837 |
| HTML | HTML available | 325 | 0.959 | 0.360 | 0.789 |
| Vision | unrestricted | 700 | 0.728 | 0.000 | 0.833 |
| Vision | screenshot available | 314 | 0.953 | 0.000 | 0.902 |
| Fusion | unrestricted | 700 | 1.000 | 1.000 | 1.000 |

**What this actually means:**

1. **Multimodality bought nothing measurable.** The URL branch alone is already
   perfect on this test set; fusion inherits that. There is no evidence here that
   page content helps.
2. **The HTML and vision models are not usable standalone detectors.** Specificity
   of 0.00–0.36 means they label most *legitimate* pages as phishing. Their F1
   looks respectable only because phishing is ~92 % of each restricted subset.
3. **Availability is a label giveaway.** A phishing URL yields a page 76 % of the
   time; a legitimate URL only 9 %, because the phishing hosts have expired since
   the labels were made. A model reading *only* the availability mask — never
   opening a page — scores **F1 0.834**. The fused model is mask-aware by design,
   so it can take that shortcut.

This confound is a property of the benchmark and cannot be fixed by collecting
more pages from the same URLs. Evidence: `backend/reports/availability_confound.md`
and `backend/reports/availability_controlled_metrics.md`.

## What is real right now

| Area | State |
| --- | --- |
| Environment inspection | done — `backend/reports/environment_summary.md` |
| Dataset inspection report | done — `backend/reports/dataset_inspection.md` (real numbers) |
| Grouped split + indices | done — `backend/data/splits/`, zero domain overlap |
| Config + structured logging | done |
| **URL model (P2)** | **done — trained, calibrated, evaluated** |
| **Leakage experiment (P2)** | **done — `backend/reports/leakage_experiment.md`** |
| **Multimodal collection (P3)** | **done — 2,100 attempted, 1,001 HTML, 970 both** |
| **HTML / vision / fusion (P4-P6)** | **done — trained; see the negative result above** |
| **Explainability (P7)** | **done — Integrated Gradients, screenshot saliency, leave-one-out** |
| **FastAPI service (P8)** | **done — `/health`, `/metrics`, `/analyze`, SSRF guard, rate limit** |
| **React frontend (P9)** | **done — no login, modality toggles, 20 tests pass** |
| **Docs + limitations (P10)** | **done — `backend/docs/`** |

Test status: **299 backend tests pass**; frontend `npm run test` (20), `typecheck`,
and `build` all pass.

## The blocker that remains

**Docker is not installed.** The isolated Chromium screenshot service could not be
built or verified. Screenshot capture therefore runs **in-process** with reduced
isolation — no filesystem, PID, or network namespace around the browser. This is
reported honestly rather than described as sandboxed: `GET /health` returns
`screenshot_service: "in_process"`, and the deployment checklist is in
`backend/docs/security.md` §7.

Also not run: the `full` training profile (CPU-only; ~161 min/epoch estimated) and
the DeiT-tiny backbone.

## Honest environment constraints

This host is a low-power laptop with **no usable GPU for deep learning**:

- Intel i5-1035G1, 4 cores / 8 threads
- 11.8 GB RAM, frequently under 2 GB free
- GeForce MX330 (2 GB) present, but PyTorch is the **CPU-only** build, so
  `torch.cuda.is_available()` is `False`
- Docker not installed

This is why every config has a `dev` and a `full` profile with **identical
architecture** and different capacity only. Measured on this host: **0.37 s per
training step**, so the `full` profile's 164,760 rows would take roughly **161
minutes per epoch**, and the DeiT-tiny vision training in P5 is not practical
here at all. A hardware fact, not a modelling choice.

## Layout

```
backend/
  app/{api,schemas,security,services,models,preprocessing,core,utils}
  training/     inspect_dataset.py, make_splits.py, train_*.py, evaluate.py
  tests/        pytest suites
  configs/      dev.yaml, full.yaml
  data/         git-ignored raw data, collected artifacts, splits
  checkpoints/  git-ignored trained weights
  reports/      committed research record (metrics, tables)
  docs/         prose, derived from reports/
  docker/       API + hardened screenshot service
frontend/       React + Vite + Tailwind dashboard
```

## Install

```powershell
# Backend (Python 3.10)
cd backend
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt `
    --extra-index-url https://download.pytorch.org/whl/cpu
Copy-Item .env.example .env      # then edit PHIUSIIL_CSV / MODEL_DIR

# Frontend (Node 20+)
cd ..\frontend
npm install
```

## Data policy

- Collected HTML and screenshots are **never committed** (`data/` is ignored).
- PhiUSIIL contains URL-level rows plus page-derived engineered columns. It
  contains **no raw HTML and no screenshots** — those are collected separately
  by the P3 acquisition pipeline.
- The URL model consumes only the raw URL string plus an explicitly documented,
  separately-ablated handcrafted branch. Training on the dataset's engineered
  columns is run only as a labelled leakage experiment.

## Documentation

| Doc | Contents |
| --- | --- |
| `backend/docs/architecture.md` | Design, and an explicit as-built section listing where the implementation diverged from the P1 proposal |
| `backend/docs/run.md` | Every command, in order, plus reproducibility and the stated environment limits |
| `backend/docs/security.md` | Threat model, SSRF defence, renderer isolation posture, deployment checklist |
| `backend/docs/limitations.md` | **What the results do not show** — read this before quoting any number |

All numbers quoted in the docs trace to a committed file in
`backend/reports/`. Where a claim could not be verified, it is marked as such
rather than asserted.

## License note

PhiUSIIL is redistributed by its authors for academic use; its terms apply to
any derived artefact. Pretrained backbones carry their own licences. The
software in this repository is provided for academic research.
