# Claims vs. implementation

Audit of every claim in the presentation against the code as it actually exists.
Written so that a marker comparing the deck to the repository finds no
discrepancy, and so that anything we cannot evidence is removed or reworded
rather than defended.

Three categories:

- **Implemented** — built, and reachable from the running system.
- **Partially implemented** — the idea is real, but the current wording
  overstates it. The reworded column is what the slide should say.
- **Not implemented** — absent from first-party code. Do not claim.

Searches were run across `Backend/` and `Frontend/` excluding `node_modules`,
`.venv`, `dist`, and data directories. Every "not implemented" row below was
confirmed by an empty search for the corresponding symbol.

---

## 1. Implemented

| Claim | Where it lives |
|---|---|
| URL character model (CNN + BiLSTM) | `app/models/url_model.py`, trained by `training/train_url.py` |
| HTML feature model | `app/models/html_model.py` (MLP over 45 DOM/text features) |
| Screenshot model | ResNet-18 over rendered page screenshots |
| Mask-aware / availability-aware fusion | `app/services/inference.py:250` (`InferenceService.fuse`) |
| Live HTML and screenshot acquisition | Playwright Chromium, `app/services/screenshot.py` |
| Explainable AI | `app/services/xai.py` — integrated gradients for URL, HTML, and vision |
| Calibration | `TemperatureScaler` fit in `training/train_multimodal.py`; loaded at `inference.py:143` |
| REST API | `app/api/main.py` — `/analyze`, `/health`, plus static and docs routes |
| React dashboard, no login | `Frontend/` |
| Domain-grouped train/val/test split | `app/preprocessing/splits.py` |
| Unicode/IDN hardening | `app/preprocessing/url_preprocessing.py` (`fold_confusables`) |
| Adversarial robustness evaluation | `training/adversarial_eval.py` → `reports/adversarial_robustness.md` |
| Reproducible notebooks | `Backend/notebooks/01..06_*.ipynb` |

Note the split: this table is *trained models and core pipeline only*. The
certificate / JS / layout / brand-image extractors are real but unsupervised, so
they belong in §2b rather than here.

---

## 2. Now implemented since the first audit

The seven previously-missing items. Each entry states what was built, and — just
as importantly — what it is and is not allowed to be claimed as.

### 2a. Fully trained models with real measured metrics

**GNN over domain–IP relationships** — `app/models/graph_model.py` (hand-written
`GraphConv`, no `torch_geometric`), `app/preprocessing/graph_build.py`,
`training/train_gnn.py`. Checkpoint `checkpoints/graph_model.pt`.

| metric | GCN | majority baseline |
|---|---|---|
| accuracy | 0.6853 | 0.6406 |
| balanced accuracy | 0.6338 | 0.5000 |
| MCC | 0.2862 | 0.0000 |
| ROC-AUC | 0.6783 | 0.5000 |

**Claim this carefully.** It beats a constant baseline, so the message passing
learns something, but it is nowhere near the URL branch. And the measured
limitation is severe: **0 of 6512 edges are domain–IP.** With no DNS resolution
the graph contains no hosting information at all — every edge is
domain↔domain structural (shared TLD, shared subdomain, containment). The
"domain–IP" claim is only true of the module's design, not of the graph we built.

**Continual learning** — `app/services/continual.py` (reservoir buffer,
class-balanced sampler, diagonal-Fisher EWC), `training/continual_update.py`.
Simulated daily stream of newly-observed evasive URLs.

| | protected (rehearsal+EWC) | unprotected baseline |
|---|---|---|
| Stream F1, final round | **0.9981** | 0.9516 |
| Clean holdout F1, final | 0.9966 | 0.9933 |

**Claim half of this.** Rehearsal + EWC clearly bought *adaptation* — the
unprotected arm never recovers after round 2. But **forgetting was not
demonstrably mitigated**: the arms differ by 0.0034 F1, probability drift was
actually *higher* with protection (0.0046 vs 0.0040), and the base model was
already at F1 0.9995 on the full test set, so there was no headroom to lose. The
report says "inconclusive" in those words. Do not upgrade that to "solved".

### 2b. Implemented, wired and tested — but no trained head

These are real: real extraction from a real observation, exposed through the API
with availability flags and reasons, 96 backend tests covering them. What they
do **not** have is a classifier trained on them, and that gap must be stated.

| Signal | Module | Features | Status |
|---|---|---|---|
| SSL/TLS certificate | `app/preprocessing/cert_features.py` | 23 | Real: live TLS probe, DER walk, self-signed fallback |
| JavaScript behaviour | `app/preprocessing/page_analysis.py` | 25 | Real: `add_init_script` runtime instrumentation, not tag counting |
| Login page design | `app/preprocessing/page_analysis.py` | 20 | Real: DOM geometry, overlap, centring, below-fold, click targets |
| Logo / brand impersonation | `app/preprocessing/page_analysis.py` | 17 | Real: brand token in `src`/`alt`/`title` vs. page host |

**Why there is no trained head.** These come from a live observation of the
target. We have no labelled crawl carrying these features — PhiUSIIL is URL
strings, and the 2,100-row page manifest has no certificate, runtime or geometry
labels. Producing a trained head needs a fresh labelled crawl, which is days of
work, not hours.

So `/analyze` returns them under a `signals` block, and the UI labels them
**"observed, not scored"**. That label is deliberate: they sit beside the
modality scores, and without it a reader would assume they contribute to the
verdict. They do not. Only the graph row has a trained head.

Note the certificate work includes a detail worth knowing: `getpeercert()`
returns `{}` under `CERT_NONE`, which is exactly the path most phishing hosts
take. A plain implementation would return all-zeros for precisely the cases it
exists to study, so the module carries a small DER walker over
`getpeercert(binary_form=True)`.

### 2c. Adversarial training

`training/train_url.py --adversarial-augment`, using `augment_for_training` in
`training/adversarial_eval.py`. Applies the semantic evasion families to the
**training split only** — val and test are never perturbed, because perturbing
them would leak the attack families into evaluation and make the robustness
numbers meaningless.

Two design points that matter more than the augmentation itself:

1. **Both classes are perturbed, in proportion.** Perturbing only phishing URLs
   teaches the inverse lesson — that surface mangling signals phishing — and
   makes the model *more* brittle.
2. **Clean originals are kept.** Perturbed copies are added, never substituted,
   so augmentation widens coverage instead of quietly deleting the distribution.

Measured run: 40,000 → **61,514** training rows (21,514 added, ratio 0.6,
families `brand_swap`, `subdomain_prepend`, `typosquat`, `path_shuffle`,
`double_encode`, `repeat_pad`). **Result in §6: evasion went to 0.000 on all
eleven families with no accuracy cost.** The normaliser-handled families
(homoglyph, fullwidth, zero-width, scheme-case, trailing-dot) are excluded from
the augmentation set because the normaliser already neutralises them, so
duplicating them would only dilute the attacks that actually matter.

---

## 3. Still partially implemented — reword before presenting

| Original claim | Problem | Say this instead |
|---|---|---|
| "Multimodal fusion" | Fusion only engages when ≥2 modalities are available. Phishing URLs fail live fetch 76% of the time, so many analyses are effectively URL-only. | "Fusion engages across whichever modalities are available; availability is reported explicitly per request." |
| "99.9% accuracy" | Real, but it is a PhiUSIIL held-out test score with domain-grouped splitting. It measures separability on one benchmark, not real-world accuracy. | "99.95% F1 on the held-out PhiUSIIL test split, with disjoint train/test domains." |
| "Explainable AI" | URL attribution was the only explanation wired into `/analyze` until the hardening round. HTML, vision, and fusion explanations existed but were unreachable. | "Integrated-gradient attribution per modality, plus a leave-one-modality-out ablation." **Now true.** |
| "Adversarial testing" | No adversarial evaluation existed before the hardening round. | "Eleven documented evasion families, with measured evasion rates, plus adversarial training on the semantic families." **Now true.** |
| "Multi-modal HTML + screenshot analysis" | HTML F1 0.9588 (specificity 0.36) and vision F1 0.9533 (specificity 0.00, MCC 0.00). Both are recall-dominated: vision alone classifies *everything* as phishing and still scores F1 0.95. | "Both single-modality models are recall-heavy; the vision head currently predicts phishing for every input and needs further training." |

**Two rows were deleted deliberately.** "Risk score" and "Real-time risk score"
are gone from this table because they remain absent, and they are the two claims
most likely to be challenged: no `risk_score` field exists anywhere in the
codebase, and the probability is a per-request model output, not a continuously
updated score. If either survives in the deck, the safe phrasing is
"per-request phishing probability".

**On the vision head specifically:** specificity 0.00 and MCC 0.00 mean it
separates nothing. Its F1 of 0.9533 is an artifact of the phishing-positive
majority in that evaluation slice. Quoting it as evidence the screenshot model
works would not survive a marker asking one follow-up question.

---

## 4. Still not implemented — remove from the deck

Re-verified by search after the implementation round. These five remain absent
from `Backend/` and `Frontend/`:

| Claim | Searched for | Result |
|---|---|---|
| File upload | `UploadFile`, `multipart`, `FormData`, `type="file"` | No matches |
| Risk score field | `risk_score` | No matches |
| Zero-day testing | — | Not a real capability. "Tested on URLs never seen before" is already covered by the domain-grouped split; do not claim zero-day detection |
| Trained heads for the live signals | — | Certificate / JS-behaviour / layout / brand-image features are extracted and served but **no classifier consumes them**. See §2b. Claiming these as modelled inputs is the single easiest way to be caught |
| Live DNS for the graph | — | The GCN is real, but the graph has 0 domain–IP edges because nothing resolves hostnames. See §2a |

Everything else from the previous audit moved into §2. If the deck still says
"we add a GNN for domain/IP analysis", the defensible version is "we built a
graph model over domain structure; it beats a constant baseline, and feeding it
real DNS resolution is the next step."

---

## 5. The most important caveat

**Availability is confounded with the label.** In the collected multimodal
manifest, the availability rate is:

- phishing URLs: **76.4%**
- legitimate URLs: **9.1%**
- gap: **+0.6725**

A mask-only classifier — one that reads *which modality was available* and
nothing else — reaches F1 **0.834** (precision 0.918, recall 0.764).

This matters more than the headline F1 of the fusion model. The models that see
a live page are, on this dataset, mostly seeing phishing pages, so an
availability-aware fusion head can score well by learning the collection bias
rather than the phishing signal. This is a property of how the dataset was
gathered, not of the models.

**Do not present the fusion result as independent evidence.** Present it, then
present this number next to it. A reviewer who notices the confound unprompted
will trust nothing else in the deck; one who is shown it will trust the rest.

---

## 6. The total failure mode — found, diagnosed, and fixed

This is the strongest result in the project, and it is a full before/after.

**Before** (`reports/adversarial_robustness.md`, baseline model, n=1000, 500
phishing / 500 legitimate, held-out test split only):

| Attack | Evasion on phishing | Post-attack F1 |
|---|---|---|
| `brand_swap` | **1.000** | 0.0000 |
| `subdomain_prepend` | 0.302 | 0.8212 |
| `repeat_pad` | 0.040 | 0.9797 |

`brand_swap` defeated the model completely — F1 went to zero. The root cause is
a dataset artifact, measured directly:

| Suffix | Rows | Phishing rate |
|---|---|---|
| `.cf` | 1205 | 0.0000 |
| `.gq` | 493 | 0.0000 |
| `.weeblysite.com` | 3079 | 0.0000 |
| `.web.app` | 5718 | 0.0000 |
| `.ml` | 994 | 0.0010 |
| `.ga` | 1112 | 0.0045 |
| `.tk` | 209 | 0.0191 |

PhiUSIIL ties cost-free domains to legitimacy, so the model learned "cheap
domain = safe". An attacker registers on any free TLD and is not examined
further. The information is absent from the features, so no amount of tuning
would have fixed it.

**After** adversarial training (`--adversarial-augment 0.6`, 40,000 → 61,514
training rows, semantic families only). Full 5-epoch run, n=1000, same protocol,
same seed. Report: `reports/adversarial_robustness_adv_trained.{json,md}`.

| Attack | Evasion before | Evasion after | Flip rate after | F1 after |
|---|---|---|---|---|
| `brand_swap` | 1.000 | **0.000** | 0.029 | 0.9709 |
| `subdomain_prepend` | 0.302 | **0.000** | 0.001 | 0.9980 |
| `repeat_pad` | 0.040 | **0.000** | 0.000 | 0.9990 |
| all others | 0.000 | 0.000 | 0.000 | 0.9990 |

**Evasion is 0.000 on every one of the eleven families.** Clean baseline F1 went
0.998004 → 0.999001, so hardening cost nothing in accuracy. Full held-out test
split (35,305 URLs): F1 0.999333, ROC-AUC 0.999842, MCC 0.998438 — within
0.0002 F1 of the unaugmented model (0.999531), i.e. indistinguishable.

The remaining `brand_swap` flip rate is now *false alarms on legitimate URLs*
(0.058), not evasions. The model has learned that a brand-flavoured host is
suspicious, which costs a little precision on genuinely legitimate brand-hosted
pages and buys complete immunity for the attack.

Two things follow, and both are worth saying out loud:

1. **The attack worked, and we caught it.** We shipped an audit that found a
   complete evasion, diagnosed the cause to a dataset label prior, and then
   closed it. That arc is the project.
2. **Be precise about what was fixed.** This is fixed *for these families*, on
   this benchmark. An attacker who finds a surface pattern outside the augmented
   set is not covered. Adversarial training is a whack-a-mole lever, and the
   honest claim is "measured across 11 documented families, not robust by
   construction." The free-TLD prior in the training data still exists; we
   compensated for it, we did not remove it.

**One caveat about the artifact:** `reports/metrics_url_adv.json` was written by
the run that produced this checkpoint, and at that time the metrics writer did
not yet record `adversarial_augment_ratio`, so the file alone does not say the
model was adversarially trained. That is fixed in `training/train_url.py`; the
numbers above come from the training log and the robustness report, which are
unambiguous.

---

## 7. Suggested framing for the demo

1. Lead with the pipeline working end to end on a live URL. That is genuinely
   strong and fully implemented.
2. Show the domain-grouped split when discussing accuracy, so the 99.95% F1 is
   understood as a benchmark number.
3. When showing the fusion result, show the availability confound immediately
   after. Volunteering the weakest point is what makes the rest credible.
4. Close on the `brand_swap` arc: we measured a 100% evasion, diagnosed it to a
   free-TLD label prior in PhiUSIIL, and closed it with adversarial training.
   A found-then-fixed failure with a measured cause is worth more than the
   headline accuracy, and it is the part most likely to score well under
   questioning.
5. Keep the honesty boundaries intact: the GNN beats a constant baseline but is
   weak and has zero DNS edges; the certificate / JS / layout / logo signals are
   observed but not scored; continual learning improved adaptation but did not
   demonstrably prevent forgetting. State each of these before you are asked.