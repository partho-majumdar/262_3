# Architecture Proposal (P1)

Status: **proposal, approved at the P1 gate**. Nothing in this document is a
measured result. Every performance claim in the project will come from a JSON
file under `backend/reports/` produced by a real run.

---

## 1. Why this shape

The pipeline has three genuinely different views of one URL:

| Modality | Signal | Cheap? | Degrades when |
| --- | --- | --- | --- |
| URL string | Lexical cues: homograph subdomains, punycode, credential-shaped paths, keyword stuffing | Yes, always | Attacker obfuscates the string |
| HTML | Structural cues: external form action, hidden iframe, obfuscated inline JS, brand-in-title mismatch | No, needs a live fetch | Site is dead (very common for phishing) |
| Screenshot | Visual cues: brand impersonation, credential-harvest layout, urgency banners | No, needs a live render *and* JS execution | Site is dead, or blocks the render |

The three fail in different situations, which is the entire justification for
fusion. A URL-only model is always available; HTML and screenshot are not. So
the fusion layer has to be **mask-aware by construction**, not bolted on.

## 2. Module layout

```
backend/
  app/
    api/            FastAPI routers, request/response wiring (no logic)
    schemas/        Pydantic contracts mirroring the API responses
    security/       URL validation, SSRF defence, IP blocklists, redirect policy
    services/       Orchestration: the analysis pipeline, collector, history
    models/         url_model, html_model, vision_model, multimodal_fusion, xai
    preprocessing/  url normalisation/tokenisation, html tokenisation, image prep
    core/           settings, structured logging, device selection
    utils/          small pure helpers
  training/         train_url, train_html, train_vision, train_fusion, evaluate
                    + inspect_dataset, make_splits, collect_multimodal
  tests/            pytest suites (security, preprocessing, models, API, splits)
  data/             git-ignored: raw dataset, collected HTML/screenshots, splits
  checkpoints/      git-ignored: trained weights
  reports/          committed: the research record (metrics JSON, tables)
  docs/             committed: prose, derived from reports/
  docker/           Dockerfile, screenshot-service.Dockerfile, compose
frontend/           React + Vite + Tailwind dashboard
```

Rule: routers never contain model or parsing logic; models never touch the
network or the filesystem; security code is importable without torch so the
SSRF test suite runs fast and in isolation.

## 3. Model designs

### 3.1 URL model — `app/models/url_model.py`

```
normalise -> char ids (fixed vocab + UNK) -> Embedding(64)
          -> Conv1d kernels {3,5,7} (parallel) -> concat
          -> BiLSTM(96, 1 layer) -> additive attention pooling
          -> url_embedding (128) -> MLP head -> logit
```

* `MAX_URL_LENGTH` is derived from the dataset's p99 length, not guessed.
* The handcrafted branch (length, dot/hyphen/digit counts, subdomain depth,
  IP-in-host, HTTPS flag, suspicious keywords, Shannon entropy of the host) is
  a **separate module** whose output is concatenated explicitly at the pooling
  stage. It is optional and off-by-default in the headline config, and its
  contribution is reported as its own ablation row.
* Returns `url_embedding (B,128)`, `logit (B,)`, `attention (B,L)`.

### 3.2 HTML model — `app/models/html_model.py`

Static parse only. No JS execution, ever. The document is reduced to a token
stream of indicator tokens (`<tag:form>`, `attr:action=external`,
`input:type=password`, `iframe:hidden`, `script:external`, …) followed by a
capped visible-text snippet, then a small Transformer encoder (2 layers, 4
heads) with masked-mean pooling -> `html_embedding (B,128)`.

Every count is also emitted as a plain `html_features` dict, because that dict
is what the XAI layer narrates.

### 3.3 Vision model — `app/models/vision_model.py`

Pretrained backbone, fine-tuned, with a 128-d projection head.

| Config | Backbone | Input | Notes |
| --- | --- | --- | --- |
| `dev` | ResNet18 (ImageNet) | 224x224 | Grad-CAM; trains on CPU in minutes |
| `full` | DeiT-tiny (ImageNet) | 224x224 | Attention rollout; needs a GPU to be practical |

Architecture is identical across configs; only the backbone and the
explanation method change. Grad-CAM for CNN, attention rollout for ViT.

### 3.4 Adaptive fusion — `app/models/multimodal_fusion.py`

Per-modality LayerNorm projection to shared dim `d=128`. With confidence
features `c_m` (unimodal logit, entropy):

```
s_m  = w^T tanh(W [e_m ; c_m])          s_m = -inf where a_m = 0
alpha = softmax(s)                       sums to 1 over available modalities
z     = sum_m alpha_m * P_m(e_m)         only over available m
logit = MLP(z)
```

Trained with modality dropout (each modality independently dropped with p, so
the all-but-one case is *learned*), auxiliary unimodal heads, and class
weighting. The `alpha` vector is returned to the UI — it is the only fusion
information displayed, and it is learned per sample, never hardcoded.

Baselines implemented and evaluated: concatenation MLP, probability average,
fixed learned-scalar weighted average. The full formulation lives in
`docs/fusion.md`.

### 3.5 Calibration

Temperature scaling fitted on the validation split, compared against Platt and
isotonic on Brier score and ECE. **Definition used everywhere:** for calibrated
phishing probability `p`, `confidence = max(p, 1 - p)`, reported alongside the
method and the fitted parameter so the UI tooltip can state exactly what it is.

## 4. Explainability (all from the real models)

| Modality | Method | Output |
| --- | --- | --- |
| URL | Captum Integrated Gradients on the char embedding table, baseline = zeros | per-character importance + attention weights as a secondary view |
| HTML | Integrated Gradients over the indicator-token embeddings | attribution per token, mapped to human-readable indicator names |
| Screenshot | Grad-CAM (CNN) / attention rollout (ViT) | base64 heatmap overlay |
| Fusion | learned `alpha` + leave-one-modality-out probability delta | both returned |

Sentences are generated from actual attribution magnitudes and feature
thresholds, and each carries the method name that produced it. No canned text.

## 5. Security design (summary; full threat model in `docs/security.md`)

Every submitted URL is treated as hostile input:

1. Scheme allowlist (`http`, `https` only).
2. Structural rejections: `file:`, `ftp:`, `javascript:`, `data:`, `gopher:`,
   userinfo tricks (`http://good.com@evil`), over-long input, malformed hosts.
3. **We** resolve DNS, and validate *every* returned A and AAAA record against
   the blocklist: loopback, private, link-local (incl. 169.254.169.254),
   CGNAT, multicast, reserved, unspecified, IPv4-mapped IPv6, decimal/hex/octal
   encodings, `fd00:ec2::254`, `metadata.google.internal`.
4. The connection is made to the *validated IP*, with the original `Host` and
   SNI, which is what defeats DNS rebinding.
5. Every redirect hop is re-validated from step 1, with a hard hop cap.
6. Streaming download with a hard byte cap, a content-type allowlist, and
   restricted ports.
7. HTML is parsed statically; downloaded JS is never executed on the API host.
8. Screenshots run in a separate, network-internal container (see
   `docker/screenshot-service.Dockerfile` for the ten defence layers).
9. Per-client rate limiting on `POST /api/analyze`.

## 6. Dependency plan

Full, version-pinned list with one-line reasons: `backend/requirements.txt`
and `frontend/package.json`. Both are verified by an actual install, recorded in
`reports/environment_summary.md`.

Pinning rationale: reproducibility. An unlocked ML stack silently changes
results between runs, which would make every number in `reports/` unverifiable.

## 7. dev vs full configs

Required because this host has no usable GPU. Both configs share one
architecture definition; only capacity knobs differ.

| Knob | `configs/dev.yaml` | `configs/full.yaml` |
| --- | --- | --- |
| epochs | 5 | 40 |
| batch size | 32 | 128 |
| CNN channels | 48 | 128 |
| LSTM hidden | 48 | 128 |
| Transformer layers (HTML) | 2 | 4 |
| Vision backbone | ResNet18 | DeiT-tiny |
| vision image size | 224 | 224 |
| max train rows | 40,000 | unlimited |
| modality dropout | 0.3 | 0.2 |

## 8. Open blockers at the P1 gate

See `reports/environment_summary.md` for measured evidence. Two items need a
decision before later phases can complete:

1. **The PhiUSIIL file is not present on this machine.** It must be supplied;
   the inspection and split scripts refuse to run without a real file rather
   than fabricating a dataset.
2. **Docker is not installed.** The sandboxed screenshot service cannot be
   built or verified here. Until it is available, screenshot capture must be
   treated as unavailable rather than silently run unsandboxed.

> **Resolution at the P10 gate.** (1) was resolved - the real file is present and
> every split number comes from it. (2) was **not** resolved: Docker is still
> absent. Screenshots are captured in-process with the browser hardened from the
> inside, and the residual risk is recorded in `reports/collection_report.md`
> rather than being described as sandboxed.

---

# As built (P3 - P10)

The rest of this document is what was *proposed* at P1. This part is what was
actually written, including the places the implementation diverged.

## 9. Module layout as built

| Concern | Module |
| --- | --- |
| SSRF / open-redirect guard | `app/security/url_guard.py` |
| Hardened HTTP fetching | `app/services/fetcher.py` |
| Screenshot capture | `app/services/screenshot.py` |
| HTML DOM features | `app/preprocessing/html_features.py` |
| Visible-text tokenizer | `app/preprocessing/html_tokenizer.py` |
| Manifest join + modality masks | `app/preprocessing/multimodal_dataset.py` |
| HTML modality model | `app/models/html_model.py` |
| Vision modality model | `app/models/vision_model.py` |
| Mask-aware fusion | `app/models/fusion_model.py` |
| Attribution | `app/services/xai.py` |
| Orchestration | `app/services/inference.py` |
| HTTP surface | `app/api/main.py` |
| Availability-confound analysis | `training/availability_confound.py` |
| Mask-controlled evaluation | `training/evaluate_controlled.py` |

## 10. Divergences from the P1 proposal

| Proposed | Built | Why |
| --- | --- | --- |
| `app/models/multimodal_fusion.py` | `app/models/fusion_model.py` | Naming consistency with the other model modules. |
| Separate containerised screenshot service | In-process Chromium, hardened | Docker is not installed on this host. The container remains the deployment target; see §11. |
| Transformer encoder over HTML tokens | Implemented as proposed | 4096-token vocabulary fitted on the pages that were collectable. |
| DeiT-tiny vision backbone (`full`) | ResNet-18 (`dev`) | As configured. The `full` profile's DeiT-tiny was not run on this host. |
| Pretrained weights assumed available | Falls back to random init and records it | `VisionModalityModel.pretrained_loaded` is written into the checkpoint and the report, so a randomly initialised backbone is never presented as pretrained. |

## 11. Isolation posture for page rendering

The container is the deployment contract. Without Docker the renderer is
hardened in-process, which is a real reduction in posture and is worth stating
precisely:

**Provided in-process:** a fresh browser context per page; every non-`http(s)`
scheme aborted at the request layer (so a page cannot read `file://` or exfiltrate
over `ftp://`/`ws://`); all permissions denied; JS dialogs auto-dismissed;
downloads refused; hard navigation timeout; total response byte cap; hardened
Chromium flags.

**Not provided:** no filesystem, PID, or network namespace around the browser.
A Chromium vulnerability would execute with the collector's privileges. The
containerised service is the only form suitable for untrusted traffic.

## 12. No authentication, and what that means

There is no login, per the project brief. The security consequences are therefore
carried entirely by two controls rather than by identity:

1. the SSRF guard, applied to every submitted URL **before** any network access
   and re-applied at every redirect hop; and
2. a fixed-window rate limit on `/analyze`, which bounds how fast the service can
   be used as a fetch proxy.

Both are tested adversarially (`tests/test_url_guard.py`, `tests/test_api.py`).
The consequence to be aware of: **anyone who can reach this service can make it
fetch URLs on the server's network.** A reverse proxy is required before it is
exposed to an untrusted network. This is stated in `/health` and in the UI, not
left implicit.
