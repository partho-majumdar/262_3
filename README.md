# Multimodal Phishing URL Detector

A phishing URL detector that looks at a link from three angles: the raw URL string, the page's HTML, and a rendered screenshot. A mask-aware gating network combines the three, and explanations are computed from the trained models themselves rather than written from templates.

The project was built and evaluated on the [PhiUSIIL](https://archive.ics.uci.edu/) dataset. The headline accuracy is very high, but the more useful part of this work is what the evaluation exposed about the dataset and about the model. Please read [Key findings](#key-findings) and [Limitations](#limitations) before quoting any number from this repository.

All figures below come from runs on the development machine described in [Environment](#environment) and are backed by files in `backend/reports/`.

---

## Contents

- [Key findings](#key-findings)
- [Dataset](#dataset)
- [Models and results](#models-and-results)
  - [URL model](#url-model)
  - [Adversarial robustness](#adversarial-robustness)
  - [Unicode and IDN handling](#unicode-and-idn-handling)
  - [Multimodal fusion](#multimodal-fusion)
  - [Graph branch](#graph-branch)
  - [Continual learning](#continual-learning)
  - [Leakage experiment](#leakage-experiment)
- [Live signals](#live-signals)
- [Explainability](#explainability)
- [Getting started](#getting-started)
- [API](#api)
- [Frontend](#frontend)
- [Testing](#testing)
- [Reproducing the results](#reproducing-the-results)
- [Environment](#environment)
- [Limitations](#limitations)
- [Repository layout](#repository-layout)
- [Documentation](#documentation)
- [Data policy](#data-policy)
- [Citation and licensing](#citation-and-licensing)

---

## Key findings

Three results stand out. They point the same way: in this dataset the URL string alone nearly solves the task, and the trained model is a good PhiUSIIL classifier rather than a general-purpose phishing detector.

### 1. The URL string is enough

The URL-only model reaches an F1 of 0.99953 on the held-out test split. Adding HTML and screenshots did not measurably improve on that, and the fused model's perfect score on the multimodal test set comes from the URL branch. The HTML and vision models are not usable on their own (specificity between 0.00 and 0.36).

### 2. Page availability leaks the label

A phishing URL in this dataset returns a live page 76% of the time, against only 9% for legitimate URLs, largely because the phishing hosts have since gone offline. A model that sees only the availability mask, and never opens a page, scores an F1 of 0.834. Any architecture that is aware of availability can exploit this, ours included. It is a property of the benchmark, and collecting more pages from the same URLs would not remove it.

### 3. Verified phishing on free hosting is missed

Three URLs confirmed as phishing on PhishTank all score as legitimate:

| URL | Probability | Verdict |
| --- | ---: | --- |
| `https://centrala-administracja.vercel.app/` | 0.000009 | legitimate |
| `https://login-outlook365.yzz.me/` | 0.000017 | legitimate |
| `https://fb-meta-verified-14259.vercel.app/` | 0.000011 | legitimate |

The cause is traceable. PhiUSIIL contains roughly 12,600 rows hosted on free platforms, and every one of them is labelled legitimate:

| Registered domain | Rows | Labelled phishing |
| --- | ---: | ---: |
| `web.app` | 5,754 | 0.0% |
| `weeblysite.com` | 3,097 | 0.0% |
| `workers.dev` | 1,438 | 0.0% |
| `glitch.me` | 515 | 0.0% |
| `github.io` | 407 | 0.0% |
| `netlify.app` | 283 | 0.0% |
| `pages.dev` | 127 | 0.0% |
| `vercel.app` | 70 | 0.0% |
| `yzz.me` | not in dataset | n/a |

From 12,600 examples and no counter-examples, the model learned that free hosting means safe. That is a reasonable inference from the data it was given, and a bad one in practice, since current phishing campaigns favour these platforms because they are free and instant to deploy.

This is a dataset bias, not a bug, and retraining on PhiUSIIL cannot fix it because there are no positive examples to learn from. Fixing it needs signal from outside the corpus, such as blocklist reputation, passive DNS, or priors about abused hosting platforms.

The adversarial evaluation is what surfaced this. A plain accuracy number on the test split would have hidden it.

### Label noise

The labels are also not perfectly clean. Genuine NASA subdomains (`www.nasa.gov`, `www.jwst.nasa.gov`, `www.giss.nasa.gov`, and others) are labelled phishing, and `https://www.google.org` appears in the test split labelled phishing. The model reproduces these labels. For that reason, results here should be read as agreement with PhiUSIIL, not as real-world phishing accuracy.

---

## Dataset

`Dataset/Phishing_URL_Dataset.csv` (PhiUSIIL).

| Property | Value |
| --- | --- |
| File size | 54,160,642 bytes |
| SHA-256 | `748ae7c7…d79438ba` |
| Rows / columns | 235,795 / 55 |
| URL column | `URL` |
| Label column | `label` (1 = phishing, 0 = legitimate) |
| Class balance | 134,850 phishing (57.19%) / 100,945 legitimate (42.81%) |
| Duplicate rows removed | 425 |
| Registered domains | 175,510 (120 appear with both labels) |

Splits are made by registered domain (seed 42, fingerprint `514839cb599158da`), so a phishing URL and a legitimate URL on the same host can never land on opposite sides of a boundary. We verified that no registered domain appears in more than one split.

| Split | Rows | Domains | Phishing share |
| --- | ---: | ---: | ---: |
| Train | 164,760 | 114,190 | 0.5729 |
| Validation | 35,305 | 29,760 | 0.5729 |
| Test | 35,305 | 31,560 | 0.5729 |

Details: `backend/reports/dataset_inspection.md`

---

## Models and results

### URL model

A character-level CNN and BiLSTM with additive attention pooling runs over the normalised URL. A 30-feature handcrafted branch, computed from the URL string only, is combined with it. Because the development machine has no usable GPU, the model was trained on a 40,000-row subset of the 164,760 available training rows. The `full` profile uses the same architecture with more capacity.

Test-split results, with Platt calibration fitted on the validation split only:

| Metric | Value | 95% bootstrap CI |
| --- | ---: | --- |
| Recall (phishing) | 0.99990 | 0.99980 – 1.00000 |
| Precision | 0.99916 | 0.99877 – 0.99956 |
| F1 | 0.99953 | 0.99935 – 0.99973 |
| Specificity | 0.99887 | n/a |
| ROC-AUC | 0.99983 | n/a |
| PR-AUC | 0.99978 | n/a |
| MCC | 0.99890 | n/a |
| Brier score | 0.00053 | n/a |
| ECE | 0.00020 | n/a |

Calibration lowered ECE from 0.00263 to 0.00020. Platt scaling was chosen over temperature scaling and isotonic regression on validation Brier score.

These numbers are unusually high, and that says more about PhiUSIIL than about the model.

**Adversarially trained variant.** A second checkpoint was trained with 11 adversarial families mixed into the training stream (`--adversarial-augment 0.6`, growing 40,000 rows to 61,514):

| Metric | Shipped | Adversarial-trained |
| --- | ---: | ---: |
| F1 | 0.99953 | 0.99933 |
| Recall | 0.99990 | 0.99990 |
| Specificity | 0.99887 | 0.99834 |
| ROC-AUC | 0.99983 | 0.99984 |

Augmentation cost 0.0002 F1 and gave no measurable robustness gain. It also does not address the free-hosting blind spot: because the same 12,600 legitimate rows still dominate, this variant scores `vercel.app` phishing URLs even lower (about 5 × 10⁻⁶). We keep it as an experimental artefact and do not ship it.

Details: `backend/reports/metrics_url_adv.json`

### Adversarial robustness

`backend/training/adversarial_eval.py` applies one documented evasion technique to each held-out test URL and rescores it. The evasion rate is computed on phishing URLs only, since a phishing URL flipping to "legitimate" is the attacker succeeding. The reverse (a false alarm) is a nuisance, not a security failure.

The 11 families are `homoglyph`, `fullwidth`, `zero_width`, `scheme_upper`, `trailing_dot`, `brand_swap`, `subdomain_prepend`, `typosquat`, `path_shuffle`, `double_encode` and `repeat_pad`.

| Category | Result |
| --- | --- |
| Families handled by normalisation | mean flip rate 0.000 |
| Semantic attacks (`brand_swap`, `subdomain_prepend`) | mean flip rate 0.005 |
| Worst evasion family | `homoglyph`, 0.000 |
| Worst post-attack F1 | `brand_swap`, 0.9709 |

The zero flip rates for the Unicode families do not show that the model is robust. The normaliser resolves those perturbations before the model sees the string, so the attack never reaches it.

`brand_swap` is the weakest family for the same reason as the PhishTank misses above: it moves the host onto a cheap or free registry, where the training data contains almost no phishing.

| Suffix | Rows | Phishing rate |
| --- | ---: | ---: |
| `.web.app` | 5,718 | 0.0000 |
| `.weeblysite.com` | 3,079 | 0.0000 |
| `.cf` | 1,205 | 0.0000 |
| `.gq` | 493 | 0.0000 |
| `.tk` | 209 | 0.0191 |

The model has effectively learned that a cheap domain is a safe domain, which an attacker can arrange for under a dollar.

Details: `backend/reports/adversarial_robustness.md`, `backend/reports/adversarial_robustness_adv_trained.md`

### Unicode and IDN handling

Homograph attacks are handled at the preprocessing stage, in `backend/app/preprocessing/url_preprocessing.py`:

- Unicode NFKC normalisation.
- Cyrillic and Greek confusables (including uppercase ranges) folded to Latin.
- Fullwidth forms folded to ASCII.
- Invisible characters stripped, including U+200B, U+200C, U+200D, U+2028, U+2029, U+FEFF, U+00AD, U+180E, U+2060, U+034F, U+061C, U+202A–U+202E, U+2066–U+2069, LRM, RLM, and Khmer inherent vowels.
- Punycode left as-is, so the model sees the form that actually goes over the wire.

For example, `раypal.com` (with a Cyrillic р and а) normalises to `paypal.com`.

### Multimodal fusion

We attempted to collect 2,100 URLs. 1,001 returned HTML (47.7%) and 970 produced both HTML and a screenshot. The ResNet-18 vision backbone loaded pretrained weights (`pretrained_loaded: true`).

Because availability leaks the label, we evaluated each model twice: on the unrestricted test rows, and on a "controlled" subset where the modality exists, which holds the availability mask constant. The mask-only baseline scores F1 0.834.

| Model | Subset | n | F1 | Specificity | ROC-AUC |
| --- | --- | ---: | ---: | ---: | ---: |
| URL | unrestricted | 700 | 1.000 | 1.000 | 1.000 |
| HTML | unrestricted | 700 | 0.724 | 0.030 | 0.837 |
| HTML | controlled | 325 | 0.959 | 0.360 | 0.789 |
| Vision | unrestricted | 700 | 0.728 | 0.000 | 0.833 |
| Vision | controlled | 314 | 0.953 | 0.000 | 0.902 |
| Fusion | unrestricted | 700 | 1.000 | 1.000 | 1.000 |
| Fusion | controlled | 308 | 1.000 | 1.000 | 1.000 |

How to read this:

1. Multimodality gave no measurable benefit. The URL branch is already perfect on this test set, and fusion inherits that.
2. The HTML and vision models cannot stand alone. Specificity of 0.00 to 0.36 means they label most legitimate pages as phishing. Their F1 looks reasonable only because phishing makes up about 92% of each restricted subset.
3. A content model scoring near 0.834 F1 on the controlled subset would suggest it is reading page liveness rather than content.

When a modality is unavailable, the system says so and gives a reason. It never invents the missing evidence.

Details: `backend/reports/availability_confound.md`, `backend/reports/availability_controlled_metrics.md`

### Graph branch

A 3-layer GCN (23,382 parameters) runs over domain structure, using edges based on shared subdomain tokens, shared TLDs, and token containment.

| Metric | Value |
| --- | ---: |
| F1 | 0.7688 |
| ROC-AUC | 0.6783 |
| Recall / Specificity | 0.8169 / 0.4508 |
| MCC | 0.2862 |
| F1 vs. majority-class baseline | −0.0121 |

This branch is weak. It beats a majority-class baseline on accuracy and MCC, but its F1 is lower, and recall at a 1% false-positive rate is only 0.020.

The domain-to-IP relationships originally planned for this branch are not implemented. There is no live DNS resolution, so the graph contains no infrastructure edges:

```
edge_type_counts.domain_ip                 = 0
graph_honesty.infrastructure_edges         = 0
fraction_of_nodes_with_infrastructure_edge = 0.0
```

All 6,512 edges are domain-to-domain name-structure edges, so this is an experiment on URL naming structure and not a model of domain/IP infrastructure. The report identifies passive or historical DNS data as the most valuable improvement.

Details: `backend/reports/metrics_gnn.json`

### Continual learning

A rehearsal buffer (capacity 600) combined with EWC streams the 11 adversarial families through the model over 12 rounds, compared against an unprotected baseline.

| Outcome | Protected | Unprotected |
| --- | ---: | ---: |
| Stream F1, first round | 0.8664 | 0.8664 |
| Stream F1, final | 0.9981 | 0.9516 |
| Clean-holdout F1, final change | +0.0017 | −0.0017 |
| Clean-holdout F1, worst | 0.9204 | 0.9190 |

Adaptation clearly works: stream F1 rose from 0.8664 to 0.9981 with protection, against 0.9516 without it.

The forgetting results are inconclusive. The two arms differ by 0.0034 F1 on a 600-URL holdout, which amounts to a handful of individual verdicts. The run's own report says the sign of this gap is not established and should not be cited as evidence in either direction. The base model already sat near 0.9995 F1, so there was little for the protection to preserve.

Updates are run manually with a script. There is no scheduled job.

Details: `backend/reports/continual_learning.json`

### Leakage experiment

PhiUSIIL includes 31 engineered columns derived from fetching the page, information a URL-only detector would never have. As a check, we trained an MLP on those columns and compared it with the shipped model on the same test split.

| Model | Phishing recall | ROC-AUC |
| --- | ---: | ---: |
| URL-only (shipped) | 0.99990 | 0.99983 |
| Engineered-column MLP | 0.99975 | 0.99999 |
| Difference | −0.00015 | n/a |

Despite having strictly more information, the engineered-column model did not beat the shipped model on recall. The leak exists in the data but does not inflate our headline figure.

Details: `backend/reports/leakage_experiment.md`

---

## Live signals

Live probes are off by default, since they open sockets and a browser. When enabled, each signal is reported next to the verdict along with its availability and any failure reason.

| Signal | Notes |
| --- | --- |
| TLS certificate | 20 features, including chain verification, self-signed detection, SAN match, certificate age and validity span. A DER fallback parser handles untrusted certificates that OpenSSL declines to parse. |
| JavaScript behaviour | Observed through a live browser probe. |
| Page layout geometry | Observed live. |
| Brand-image impersonation | Observed live. |

These signals are currently reported only. No trained head consumes them, and only the graph branch carries a real model probability among them. The API schema (`app/schemas/analysis.py`) states this explicitly.

Code: `backend/app/services/signals.py`, `backend/app/preprocessing/cert_features.py`

---

## Explainability

Attributions are computed from the trained models.

| Method | Scope |
| --- | --- |
| Integrated Gradients | URL characters and HTML tokens |
| Saliency maps | Screenshot regions |
| Leave-one-modality-out | `modality_influence`, recomputed on every request |
| Per-modality probabilities and gate weights | Returned with every verdict |
| Missing-evidence reasons | Explains why a modality was unavailable |

These are exposed through `POST /analyze` and displayed in the frontend.

---

## Getting started

### Backend (Python 3.10)

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt `
    --extra-index-url https://download.pytorch.org/whl/cpu
Copy-Item .env.example .env   # then edit PHIUSIIL_CSV and MODEL_DIR
```

### Frontend (Node 20+)

```powershell
cd frontend
npm install
```

---

## API

Start the server:

```powershell
python -m uvicorn app.api.main:app --host 127.0.0.1 --port 8000
```

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | Model load state, screenshot service mode, calibration status, notes |
| `GET /metrics` | Served metrics |
| `POST /analyze` | Full pipeline with modality selection, live signals and explanations |

Security controls include an SSRF guard (scheme, port, redirect, response-size and per-domain limits) and per-client rate limiting at 10 requests per minute. TLS verification is intentionally disabled during analysis so that broken or self-signed certificates can still be inspected; the rationale is recorded in the source. See `backend/docs/security.md`.

**Deployment notes.** These are reported by `GET /health` but intentionally not shown in the UI:

- The service has no authentication. Anyone who can reach it can make it fetch URLs, so it should not be exposed to an untrusted network without a reverse proxy.
- Probability calibration is not active for the served fusion probability, because `checkpoints/fusion_calibration.json` was never generated. The API reports this and falls back to a raw sigmoid.

---

## Frontend

The dashboard is built with React 18, Vite and Tailwind, and talks to the real backend with no hardcoded demo results.

```powershell
cd frontend
npm run dev
```

Modalities can be toggled individually. The result panel shows the verdict, gate weights, per-modality probabilities, the leave-one-out ablation and the calibration status, with a clear warning whenever a score is not a calibrated probability. Unavailable evidence is shown as unavailable, with a reason, rather than as a zero score.

The example URLs on the page are held-out PhiUSIIL test URLs (3 legitimate, 4 phishing), each checked against the shipped checkpoint. Hand-written phishing examples are deliberately left out because of the free-hosting blind spot described above.

---

## Testing

| Suite | Result |
| --- | --- |
| Backend `pytest` | 582 passed |
| Frontend `vitest` | 30 passed |
| Frontend `tsc --noEmit` | clean |
| Frontend `build` | passes |

One backend test checks that the graph edge set is byte-identical under label permutation. This confirms the GNN cannot be reading labels through its edges.

---

## Reproducing the results

Run from `backend/`. Full details are in `backend/docs/run.md`.

```powershell
# 1. Dataset inspection and integrity checks
.\.venv\Scripts\python.exe -X utf8 training/inspect_dataset.py --csv ../Dataset/Phishing_URL_Dataset.csv

# 2. Domain-grouped splits
.\.venv\Scripts\python.exe -X utf8 training/make_splits.py `
    --csv ../Dataset/Phishing_URL_Dataset.csv --out data/splits --seed 42

# 3. URL model
.\.venv\Scripts\python.exe -X utf8 training/train_url.py --config configs/dev.yaml --tag url

# 4. Adversarially trained variant
.\.venv\Scripts\python.exe -X utf8 training/train_url.py --config configs/dev.yaml `
    --tag url_adv --no-resume --adversarial-augment 0.6

# 5. Adversarial evaluation
.\.venv\Scripts\python.exe -X utf8 training/adversarial_eval.py --ckpt checkpoints/url_model.pt

# 6. Graph model
.\.venv\Scripts\python.exe -X utf8 training/train_gnn.py --config configs/dev.yaml --tag gnn

# 7. Continual learning (rehearsal + EWC, with unprotected baseline)
.\.venv\Scripts\python.exe -X utf8 training/continual_update.py --config configs/dev.yaml --baseline

# 8. Multimodal data collection and training
.\.venv\Scripts\python.exe -X utf8 training/collect_multimodal.py --config configs/dev.yaml
.\.venv\Scripts\python.exe -X utf8 training/train_multimodal.py --config configs/dev.yaml
```

Every number in this README comes from a file in `backend/reports/`.

---

## Environment

Everything was developed on a low-power laptop:

- Intel i5-1035G1 (4 cores / 8 threads)
- 11.8 GB RAM, often with under 2 GB free
- GeForce MX330 (2 GB), unused because the installed PyTorch build is CPU-only (`torch.cuda.is_available()` returns `False`)
- Docker not installed

Each config has a `dev` and a `full` profile with the same architecture and different capacity. Training ran at about 0.37 s per step on this machine, which would put one epoch of the `full` profile (164,760 rows) at roughly 161 minutes. As a result, the `full` profile and the DeiT-tiny vision variant were not run.

The isolated screenshot service could not be built or tested here. Screenshots are captured in-process, without filesystem, PID or network namespace isolation around the browser. `GET /health` reports this as `screenshot_service: "in_process"`.

---

## Limitations

- **The model reflects PhiUSIIL's biases.** It misses phishing hosted on free platforms and mislabels some legitimate domains, because the training data does.
- **Content models are not independent detectors.** The HTML and vision models have low specificity, and availability of a page already predicts the label.
- **The graph branch has no infrastructure data.** There is no DNS resolution, so it cannot model domain and IP relationships.
- **Live signals are not used in the verdict.** They are reported only.
- **Fusion probabilities are uncalibrated.** The calibration file for the fusion model has not been generated.
- **No authentication and reduced screenshot isolation.** Do not expose this service to untrusted networks as-is.
- **Training used a subset.** The shipped URL model saw 40,000 of 164,760 training rows.

A fuller discussion is in `backend/docs/limitations.md`.

---

## Repository layout

```
backend/
  app/{api,schemas,security,services,models,preprocessing,core,utils}
  training/     inspect_dataset.py, make_splits.py, train_url.py, train_gnn.py,
                train_multimodal.py, adversarial_eval.py, continual_update.py,
                leakage_experiment.py, availability_confound.py, evaluate_controlled.py
  tests/        pytest suites (582 tests)
  configs/      dev.yaml, full.yaml
  data/         raw data, splits, manifest (git-ignored)
  checkpoints/  trained weights (git-ignored)
  reports/      committed results; every number above traces here
  docs/         written documentation derived from reports/
  docker/       API and hardened screenshot service (never built here; no Docker)
frontend/       React + Vite + Tailwind dashboard
Dataset/        PhiUSIIL CSV (not redistributable under most terms)
```

---

## Documentation

| Document | Contents |
| --- | --- |
| `backend/docs/architecture.md` | Design, plus a section listing where the implementation diverged from the original proposal |
| `backend/docs/run.md` | All commands in order, reproducibility notes, environment limits |
| `backend/docs/security.md` | Threat model, SSRF defences, renderer isolation, deployment checklist |
| `backend/docs/limitations.md` | What the results do and do not show |

---

## Data policy

- Collected HTML and screenshots are never committed (`data/` is git-ignored) and are not needed to run or evaluate the system, since inference fetches pages live. They are inputs for multimodal retraining only and can be rebuilt with `training/collect_multimodal.py`.
- PhiUSIIL contains URL-level rows and page-derived engineered columns, but no raw HTML or screenshots.
- The URL model uses only the raw URL string plus a separately ablated handcrafted feature branch. The dataset's engineered columns are used only in the labelled leakage experiment.

---

## Citation and licensing

This project builds on the PhiUSIIL dataset, and its terms apply to any derived artefacts. Pretrained backbones (ResNet-18, and DeiT-tiny where referenced) carry their own licences. The code in this repository is provided for academic research.