# How this system works — end to end

A complete walkthrough for explaining the project. Written so you can answer
follow-up questions without guessing: every number here was measured on this
machine, and every claim has a file you can point at.

**If a teacher asks "does it actually work?"**, the short version is: the
server runs, `/analyze` returns a real verdict on a live URL, all three
modality branches load, all four explanation methods execute, and 571 backend
tests plus 30 frontend tests pass.

---

## 1. The one-paragraph version

We built a phishing URL detector that looks at a URL four different ways and
combines what it finds. The **URL branch** reads the raw string as characters
through a CNN + BiLSTM. The **HTML branch** fetches the live page and reads its
DOM structure. The **vision branch** renders the page in a real Chromium browser
and looks at the screenshot. A **fusion network** combines whichever of those
actually worked, and explicitly reports which ones did not. On top of that we
measure four live signals that do not have trained models behind them — the TLS
certificate, JavaScript behaviour at runtime, page layout geometry, and
brand-image impersonation — and we train a small graph network over domain
structure. The system also audits itself: we built an adversarial evaluation
that deliberately attacks the model, found that a free domain name evaded
detection 100% of the time, diagnosed why, and fixed it.

---

## 2. Repository map

```
Backend/
  app/
    api/main.py              FastAPI endpoints: /analyze, /health, /metrics
    schemas/analysis.py      Request/response contracts
    services/
      inference.py           the orchestrator: acquire -> embed -> fuse -> calibrate
      xai.py                 4 explanation methods
      fetcher.py             hardened HTTP fetch
      screenshot.py          Playwright capture
      page_probe.py          Playwright runtime/layout/logo probe
      continual.py           rehearsal buffer + EWC
      signals.py             collects cert / page / graph signals
    models/
      url_model.py           CharCNN + BiLSTM + attention
      html_model.py          token transformer + DOM-feature MLP
      vision_model.py        ResNet-18
      fusion_model.py        mask-aware fusion
      graph_model.py         hand-written GraphConv GCN
    preprocessing/
      url_preprocessing.py   normalisation, confusable folding, tokenizer
      url_features.py        30 handcrafted URL features
      html_features.py       45 DOM/text features
      cert_features.py       23 TLS certificate features
      page_analysis.py       20 layout + 25 behaviour + 17 brand-image features
      graph_build.py         domain/IP graph construction
    security/url_guard.py    SSRF guard
  training/
    make_splits.py           domain-grouped split
    train_url.py             URL model  (+ --adversarial-augment)
    train_multimodal.py      HTML, vision, fusion
    train_gnn.py             graph model
    continual_update.py      continual-learning simulation
    adversarial_eval.py      11 attack families
  reports/                   all metrics, written by the training runs
  checkpoints/               all model weights
  docs/claims_vs_implementation.md   what we may and may not claim
Frontend/                    React dashboard, no login
```

---

## 3. The data and why the split matters

**Dataset:** PhiUSIIL, `Dataset/Phishing_URL_Dataset.csv`. 235,795 rows, 55
columns. Labels: `1 = Phishing`, `0 = Legitimate`. SHA-256
`748ae7c7…438ba`.

**The split is grouped by registered domain, not by row.** This is the single
most important methodological decision in the project.

A naive random split puts the *same domain* in train and test. The model then
memorises that domain and scores near-perfectly, which measures nothing about
detection. We group by registered domain first, then split:

| split | rows | domains overlap with test? |
|---|---|---|
| train | 164,760 | no |
| validation | 35,305 | no |
| test | 35,305 | — |

**If asked "why is your accuracy so high?"** — that is the answer. It is
high partly because the task on this dataset is genuinely separable, and partly
because we refused to let the model memorise domains. The 99.95% F1 is a
*benchmark* number on one dataset, not a real-world accuracy claim.

**Leakage experiment.** The PhiUSIIL file has 24 page-derived engineered columns
(`url_chain_length`, `domain_registration_days`, …) that would not exist at
inference time. We trained a variant on them to quantify the gap, and they are
reported separately, never mixed into the headline number.

---

## 4. The four modelled branches

### 4.1 URL branch — `app/models/url_model.py`

```
URL string
  -> normalise (NFKC, confusable folding, percent-decode, lowercase host, dedupe runs)
  -> character tokenise (frozen vocab, max_len 249)
  -> embedding (60 -> 64)
  -> 3 parallel convolutions, kernel 3/5/7, 48 channels each
  -> single-layer bidirectional LSTM, hidden 48
  -> additive attention pooling (masked; padding gets -inf)
  -> concatenate with 30 standardised handcrafted features
  -> projection to 128-d embedding  ->  logit
```

**156,913 parameters.**

Two things worth explaining:

- **Why a frozen vocabulary.** The vocabulary is fitted once during training,
  saved in the checkpoint, and reloaded verbatim. If it were rebuilt from data
  at load time, the model's behaviour would depend on which URLs happened to be
  in memory when it started.
- **Why the attention mask is masked with `-inf`.** Padded positions get
  `-inf` before the softmax so they contribute exactly nothing. A fully padded
  row would produce NaN, so the code also runs `nan_to_num`.

### 4.2 HTML branch — `app/models/html_model.py`

Live page → 45 handcrafted DOM/text features + up to 512 word tokens →
2-layer transformer encoder (4 heads, ff 256) → concatenated with the 45
features → 128-d embedding.

**927,233 parameters.** Examples of the 45 features: `n_password_inputs`,
`n_form_actions_external`, `has_meta_refresh`, `brand_host_mismatch`,
`n_external_scripts`, `text_external_urls_count`.

### 4.3 Vision branch — `app/models/vision_model.py`

Playwright renders the page → 224×224 RGB → ResNet-18 (ImageNet pretrained) →
512-d → projection to 128-d.

**11,252,181 parameters** — 90% of the total, because ResNet-18 is a full CNN.

### 4.4 Fusion — `app/models/fusion_model.py`

```
url_emb (128) ──┐
html_emb (128) ─┼─> per-modality projection to shared_dim 128
vision_emb(128) ─┘         │
                            ├─ mask-aware gate  (3 masks in, 3 weights out)
                            └─ MLP [128, 64] -> logit
```

**150,663 parameters.** `forward(emb_url, emb_html, emb_vision, mask_url,
mask_html, mask_vision)`.

**The critical design point:** fusion receives an explicit **availability mask**
per modality. A modality that could not be acquired is masked to zero and its
embedding is a zero vector. The network learns what to do with absent evidence
rather than being fed a fabricated default. `modality_dropout: 0.3` drops each
modality independently during *training* so the same code path is exercised.

**Fusion only engages when ≥2 modalities are available.** With one modality
the system reports `fused: false` and returns that branch's own score.

**Total: 12,486,990 parameters** across the four branches, plus **23,382** for
the graph model.

---

## 5. The four live signals (implemented, but not scored)

These are real measurements, fully implemented and tested, and served in the
`signals` block. **They have no trained classifier head**, so they do **not**
contribute to the verdict. The UI labels them "observed, not scored". Be
straight about this if asked.

| signal | features | how it is obtained |
|---|---|---|
| **TLS certificate** | 23 | real TLS handshake, then a small DER walk |
| **JS behaviour** | 25 | JS injected *before* navigation via `add_init_script` |
| **Page layout** | 20 | `getBoundingClientRect` geometry |
| **Brand images** | 17 | brand tokens in `src`/`alt`/`title` vs. the page host |

**Two implementation details that are good answers to "how do you know it
works?"**

*The certificate:* `ssl.getpeercert()` returns an **empty dict** under
`CERT_NONE` — which is exactly the path most phishing hosts take, because their
certificates do not verify. A naive implementation therefore returns all-zeros
for precisely the cases it exists to study. We added a DER walker over
`getpeercert(binary_form=True)` to recover subject, issuer, validity, serial and
SAN anyway, and a feature `cert_from_der_parse` records when it was needed.

*The JS probe:* instrumentation is installed as a document-start init script, so
it records behaviour from the very first byte. The probe state is installed
`non-configurable` so a hostile page cannot `delete window.__probe`. We capture
native `Function` and `setTimeout`/`setInterval` *before* wrapping them —
wrapping `window.Function` silently broke `Function.prototype.toString` and
zeroed the timer-redirect count until we found it.

**Measured on a local test page:** `window_open_calls=1`, `js_errors=1`,
`keyboard_event_intercepts=1`, `keystroke_suppressions=1`, `timer_redirects=1`,
`form_action_cross_origin=1`, 2 inputs / 1 password, brand token found in `alt`.

**Why no trained head:** these need a labelled crawl. PhiUSIIL is URL strings,
and the 2,100-row page manifest has no certificate, runtime or geometry labels.
Producing that crawl is days of work and is the honest next step.

---

## 6. What happens when you call `/analyze`

Exact order. This is the flow to narrate.

```
POST /analyze {url, modalities, explain, include_signals}
 │
 1. SSRF guard (validate_target)  ── reject before any network I/O
 │     scheme ∈ {http,https}; port ∈ {80,443}; no loopback/private/link-local;
 │     DNS resolution happens here, off the event loop
 │
 2. acquire()  ── only if html or vision was requested
 │     SecureFetcher.fetch(url)  → status, bytes, final_url, redirects
 │     if not ok: record reason, mark html+vision unavailable
 │     Playwright capture       → PNG bytes
 │     (a modality that fails is EXCLUDED, never replaced by a default)
 │
 3. embed()  ── url / html / vision each to a 128-d vector
 │
 4. fuse()  ── mask-aware fusion over available modalities only
 │     unavailable + <2 available → return the single branch's own score
 │
 5. calibrate()  ── temperature scaling if the artifact exists
 │
 6. XAI  ── 4 methods, each wrapped in try/except so one failure cannot
 │          break the request
 │
 7. signals  ── cert + page probe + graph, gathered CONCURRENTLY,
 │              each degrading to {available: false, reason: ...}
 │
 8. strip private keys (any key starting with "_") so raw HTML, PNG bytes and
    embeddings never reach the JSON response
```

**Step 1 is the security boundary.** There is **no authentication**, so anyone
who can reach the service can make it fetch arbitrary URLs. The SSRF guard is
the only thing preventing use as an internal-network scanner. That is why the
guard runs *before* any fetch rather than inside it. A fixed-window rate limit
(per client) is the only other brake.

**Step 5 honest caveat:** `probability_is_calibrated` is currently **false** in
a fresh run, because `checkpoints/fusion_calibration.json` is not present. The
flag exists precisely so the UI can say so, and the UI does: an uncalibrated
score is shown as "raw score, not calibrated", because a raw sigmoid output is
not a frequency and must not be read as one.

---

## 7. The four explanation methods

All four execute. All four are wrapped in `try/except` so a failure degrades
instead of 500-ing — **and they log the failure**, because a silent fallback
returns a plausible-looking explanation that is not the method claimed.

| method | what it does | output |
|---|---|---|
| `explain_url` | integrated gradients over the **embedding output**, summed per character | `char 24: '/' = -30.83` |
| `explain_html` | integrated gradients over the 45 DOM features | `html_length = 91847.0` |
| `explain_vision` | pixel gradient, pooled to a 6×6 grid | `region 6x6 cell 1 = 0.00318` |
| `explain_fusion` | **leave-one-modality-out counterfactual** | `url: 0.49254, html: -0.24124, vision: -0.23515` |

**`explain_fusion` is the strongest one** because it is a genuine counterfactual:
each available modality is withheld and the fused score is recomputed. It
answers "what would the verdict have been without this evidence?" — a measured
change, not a gate weight.

**Why integrated gradients run on the embedding, not the characters.** Captum
interpolates the input toward a baseline, which turns integer character ids into
floats — and `torch.embedding` only accepts Long/Int indices. So we split the
model into `forward()` (embeds, then calls `forward_from_embedding()`) and
attribute in embedding space, summing across the embedding dimension to get one
number per character. This took four separate bug fixes; each is commented at
its site.

**A trap worth knowing:** the vision grid uses adaptive average pooling, not
`reshape`. The input is 224×224 and 224 is not divisible by 6, so
`reshape(6, -1, 6, -1)` raises.

---

## 8. Adversarial robustness and adversarial training

This is the strongest result in the project.

**11 documented attack families** in `training/adversarial_eval.py`:
`homoglyph`, `fullwidth`, `zero_width`, `scheme_upper`, `trailing_dot`,
`brand_swap`, `subdomain_prepend`, `typosquat`, `path_shuffle`,
`double_encode`, `repeat_pad`.

Each test URL is perturbed by one family and rescored. **Evasion rate** = the
fraction of *phishing* URLs that flip to "legitimate". A false alarm on a
legitimate URL is the reverse — bad, but not a security failure.

### Before (n=1000, 500 phishing / 500 legitimate, held-out test only)

| attack | evasion | F1 after |
|---|---|---|
| `brand_swap` | **1.000** | 0.0000 |
| `subdomain_prepend` | 0.302 | 0.8212 |
| `repeat_pad` | 0.040 | 0.9797 |
| other 8 | 0.000 | ≥0.9950 |

`brand_swap` defeated the model **completely** — F1 went to zero.

### Root cause, measured directly

| suffix | rows | phishing rate |
|---|---|---|
| `.cf` | 1205 | 0.0000 |
| `.gq` | 493 | 0.0000 |
| `.weeblysite.com` | 3079 | 0.0000 |
| `.web.app` | 5718 | 0.0000 |
| `.ml` | 994 | 0.0010 |
| `.ga` | 1112 | 0.0045 |
| `.tk` | 209 | 0.0191 |

PhiUSIIL marks cost-free domains as essentially 0% phishing. The model learned
**"cheap domain = safe"**. An attacker registers on any free TLD and is never
examined further. The information is not in the features, so no amount of
tuning could fix it.

### After adversarial training

`train_url.py --adversarial-augment 0.6` → 40,000 → **61,514** training rows.

| attack | evasion before | evasion after | F1 after |
|---|---|---|---|
| `brand_swap` | 1.000 | **0.000** | 0.9709 |
| `subdomain_prepend` | 0.302 | **0.000** | 0.9980 |
| `repeat_pad` | 0.040 | **0.000** | 0.9990 |
| other 8 | 0.000 | 0.000 | 0.9990 |

**Evasion is 0.000 on all eleven families, at no cost in accuracy** — clean
baseline F1 went 0.998004 → 0.999001.

Two augmentation rules that matter more than the augmentation itself:

1. **Both classes are perturbed, in proportion.** Perturbing only phishing URLs
   teaches the inverse lesson — that surface mangling signals phishing — and
   makes the model *more* brittle.
2. **Perturbed copies are added, never substituted.** The clean URL stays, so
   augmentation widens coverage instead of quietly deleting the distribution.

Validation and test are **never** perturbed. Perturbing them would leak the
attack families into evaluation and make the robustness numbers meaningless.

**The honest caveat:** this is fixed *for these families, on this benchmark*. An
attacker who finds a pattern outside the augmented set is not covered. The
free-TLD prior still exists in the data; we compensated for it, we did not
remove it.

---

## 9. Unicode / IDN hardening

In `app/preprocessing/url_preprocessing.py`. NFKC cannot fold Cyrillic/Greek
look-alikes because they are genuinely different letters, not compatibility
variants. We added an explicit, small, auditable confusable table plus removal
of invisible characters.

Handles: Cyrillic and Greek look-alikes (**including uppercase** — omitting
capitals let `Раypal.com` through, which is the attack the table exists to
stop), fullwidth forms via NFKC, zero-width family, word joiner, combining
grapheme joiner, Arabic letter mark, LRM/RLM, and the full bidi
embedding/override/isolate set (U+202A–E, U+2066–9).

`раypal.com` → `paypal.com`. Punycode (`xn--…`) is deliberately preserved.
Real internationalised domains exist and must not be rewritten.

**Measured:** evasion 0.000 on `homoglyph`, `fullwidth`, `zero_width`,
`scheme_upper`, `trailing_dot`, `typosquat`, `path_shuffle`, `double_encode` —
both before and after adversarial training.

---

## 10. Continual learning

`app/services/continual.py` — reservoir buffer (Algorithm R), class-balanced
sampler, diagonal-Fisher EWC. Simulated daily stream of newly observed URLs
perturbed by the attack families, cut from the **test** split (never trained
on), with a disjoint clean holdout.

| | rehearsal + EWC | unprotected baseline |
|---|---|---|
| Stream F1, final round | **0.9981** | 0.9516 |
| Clean holdout F1, final | 0.9966 | 0.9933 |

**Say only half of this.** Rehearsal + EWC clearly bought **adaptation** — the
unprotected arm never recovers after round 2. But **forgetting was not
demonstrably mitigated**: the arms differ by 0.0034 F1, probability drift was
actually *higher* with protection (0.0046 vs 0.0040), and the base model was
already at F1 0.9995 on the full test set, so there was no headroom to lose. The
report says "inconclusive" in those words.

Class balancing is by **oversampling the minority class**, not by loss
weighting — weighting still leaves the gradient direction dominated by one
class. The sampler reports the *realised* ratio so a report can show balancing
happened rather than assert it.

---

## 11. Graph neural network

`app/models/graph_model.py` — `GraphConv` written from scratch in pure PyTorch
(**no `torch_geometric`**, not installed), degree-normalised mean aggregation
`a_v = Σ h_u / max(|N(v)|,1)`, three layers, LayerNorm/ReLU/dropout, per-node
head. 23,382 parameters. Nodes: domains, IPs, /24 subnets. Seven edge types.

| metric | GCN | majority baseline |
|---|---|---|
| accuracy | 0.6853 | 0.6406 |
| balanced accuracy | 0.6338 | 0.5000 |
| MCC | 0.2862 | 0.0000 |
| ROC-AUC | 0.6783 | 0.5000 |

It beats a constant baseline, so message passing learns something. **It is not
competitive with the URL branch, and say so.**

**The severe limitation:** 0 of 6512 edges are domain–IP. With no DNS
resolution the graph contains no hosting information at all — every edge is
domain↔domain structure (shared TLD, shared subdomain, containment). "We built
a GNN for domain/IP analysis" is true of the *design*, not of the graph we
built. Feeding in passive/historical DNS is the single highest-value next step,
and `build_graph(..., extra_domains=...)` already supports it.

**No label leakage:** edges are built structurally and never read labels. There
is a test that asserts the edge set is byte-identical under permuted labels.

---

## 12. All metrics, honestly

### URL-only, held-out test split (35,305 URLs, domains disjoint from train)

| metric | value |
|---|---|
| recall | 0.9999011 |
| precision | 0.9991602 |
| **F1** | **0.9995305** |
| specificity | 0.9988725 |
| ROC-AUC | 0.9998251 |
| MCC | 0.9984 |

Adversarially-trained variant: F1 **0.999333**, ROC-AUC 0.999842, MCC 0.998438 —
within 0.0002 F1 of the clean model, i.e. indistinguishable.

### Multimodal (2,100-row page manifest, n=700 fused)

| branch | F1 | recall | specificity | ROC-AUC | MCC |
|---|---|---|---|---|---|
| HTML | 0.9588 | 0.97 | 0.36 | 0.789 | — |
| Vision | 0.9533 | 1.00 | **0.00** | — | **0.00** |
| Fusion | 1.0000 | 1.00 | 1.00 | 1.000 | 1.00 |

**Do not quote the vision F1 without this caveat.** Specificity 0.00 and MCC
0.00 mean it separates nothing — it classifies *everything* as phishing, and the
phishing-positive majority in that slice is what produces the 0.95. **Observed
live:** on `https://www.wikipedia.org/` the vision branch reports **0.944
phishing** and the HTML branch **0.929**, on a site that is obviously
legitimate. The URL branch says 0.00009 and fusion correctly returns
"legitimate" at 0.373.

That is a real demonstration risk. If a marker types a legitimate site, the
per-modality panel will show two numbers near 0.94 that are simply wrong. **The
correct framing:** those two branches are recall-heavy and under-trained; the
fusion is what makes the verdict correct. It also happens to be a good
illustration of why single-modality metrics must be read with the confusion
matrix, not the F1 alone.

### The most important caveat — availability is confounded with the label

| | availability rate |
|---|---|
| phishing URLs | **76.4%** |
| legitimate URLs | **9.1%** |
| gap | **+0.6725** |

A **mask-only classifier** — one that reads *which modality was available* and
nothing else — reaches **F1 0.834** (precision 0.918, recall 0.764).

This is a property of **how the dataset was gathered**, not of the models: the
sites that were still reachable in 2026 from a 2022 crawl are mostly the
phishing ones. A fusion head can score well by learning the collection bias
rather than the phishing signal. **Volunteer this number.** A reviewer who finds
it unprompted will discount everything else; one who is shown it will trust the
rest.

---

## 13. Likely questions, with answers

**"Why is your accuracy 99.9%? Isn't that unrealistic?"**
It is a benchmark number on one dataset, not a real-world claim. Two reasons it
is high: the task is genuinely separable on PhiUSIIL, and we split by registered
domain so the model cannot memorise domains and score itself. The real-world
evidence is the adversarial result: 100% evasion before hardening, which shows
how a single dataset artifact can produce a perfect-looking score.

**"How do you know it isn't leaking?"**
Splits are grouped by registered domain — 0 domain overlap between train and
test, asserted in the split tests. We also ran a leakage experiment on PhiUSIIL's
24 page-derived engineered columns, which would not exist at inference time, and
report that variant separately.

**"What happens if the page can't be loaded?"**
The modality is marked `available: false` with a human-readable reason and
**excluded** from fusion. We never substitute a default score for evidence we
do not have. With fewer than two modalities, `fused` is `false` and the single
branch's own score is returned.

**"Isn't this a security risk? Anyone can make your server fetch URLs."**
Yes, and we say so in `/health`. There is no authentication. The SSRF guard is
the security boundary: it rejects non-http(s) schemes, non-80/443 ports,
loopback, private, and link-local addresses, and it resolves DNS *before* any
fetch. A fixed-window per-client rate limit is the only other brake. It must not
be exposed to an untrusted network without authentication in front.

**"Why is fusion at 1.000 when the branches are much worse?"**
Because the fusion head is the one that exploits the availability confound
described in §12. Treat that number as untrustworthy until the confound is
removed by re-crawling with balanced collection.

**"What is your GNN actually doing if it has no IP edges?"**
It is aggregating over shared TLDs, shared subdomains and containment, not
hosting relationships. It beats a majority baseline (MCC 0.286 vs 0.000) so it
learnes *something* about name structure, but it is weak and not competitive.
DNS is the next step and the code already accepts it.

**"Are your certificate/JS/layout/logo signals part of the model?"**
No. They are measured live, served in the `signals` block, and labelled
"observed, not scored" in the UI. There is no trained head because there is no
labelled crawl carrying those features. Claiming they feed the verdict would be
wrong.

**"Does continual learning prevent forgetting?"**
We cannot show that. It clearly improved *adaptation* (stream F1 0.9516 → 0.9981
with rehearsal), but the forgetting comparison is inconclusive — 0.0034 F1
difference, and drift was slightly higher with protection. Our report says
"inconclusive" rather than claiming a win.

**"How is the system secured / what is the attack surface?"**
No auth, so: the SSRF guard is the boundary; rate limiting bounds abuse; the
fetcher enforces scheme/port/response-size/redirect caps; the browser runs
in-process, hardened from the inside, because Docker is unavailable on this
host — that is itself a known weakness worth naming.

**"Why not just use BERT like the paper?"**
We did not, and the reason is defensible: the URL is a short, character-level
string where a pre-trained word-piece vocabulary is a poor fit, and character
n-gram structure (`.tk`, `-login`, digit runs) is exactly the phishing signal.
We also cannot fine-tune a large BERT on a CPU-only box in the time available.

**"What would you do next?"**
In priority order: (1) re-crawl with balanced availability to kill the
confound; (2) feed DNS into the graph; (3) a labelled crawl to train heads for
the certificate/JS/layout signals; (4) fix the vision head, which currently
separates nothing; (5) train on the full 164,760 rows rather than the 40,000
dev cap.

---

## 14. Running and demoing it

```powershell
# backend
cd Backend
.\.venv\Scripts\python.exe -X utf8 -m uvicorn app.api.main:app --host 127.0.0.1 --port 8000

# frontend (separate terminal)
cd Frontend
npm install
npm run dev
```

Verify before you demo:

```powershell
# backend  -> expect: 571 passed
.\.venv\Scripts\python.exe -X utf8 -m pytest tests -q
# frontend -> expect: 30 passed
npm test ; npm run typecheck
```

**Demo order that works best:**

1. Analyse a live phishing-looking URL with all modalities — show the verdict,
   the per-modality scores, the four explanation panels, the leave-one-out
   ablation, and the three signal groups.
2. Mention the availability confound *immediately* after showing fusion.
3. Finish on the adversarial arc: 100% evasion found → diagnosed to a free-TLD
   label prior → closed with adversarial training, 0.000 on all eleven families.

**Demo risks, stated plainly:**

- Typing a **legitimate** site shows HTML and vision scoring ~0.94. Expected.
  Frame it as the recall-heavy branches, and point at the URL branch and the
  fused verdict being correct.
- The UI badge will read **"raw score, not calibrated"** because no calibration
  artifact is present. Say why: the flag is there precisely so we do not present
  an uncalibrated sigmoid as a frequency.
- `--include_signals` is off by default in the API (it opens a socket and a
  browser). The dashboard turns it on.
