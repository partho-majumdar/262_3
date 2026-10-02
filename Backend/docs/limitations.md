# Limitations

This document exists because the results look better than the evidence supports.
Each section states what was measured, what it does not show, and what would be
needed to actually establish the claim.

## 1a. Multimodality added nothing measurable here

Measured on the 2,100-URL test set (`reports/availability_controlled_metrics.md`):

| model | subset | n | F1 | specificity | ROC-AUC |
| --- | --- | ---: | ---: | ---: | ---: |
| URL | unrestricted | 700 | 1.000 | 1.000 | 1.000 |
| HTML | unrestricted | 700 | 0.724 | 0.030 | 0.837 |
| HTML | HTML available | 325 | 0.959 | 0.360 | 0.789 |
| Vision | unrestricted | 700 | 0.728 | 0.000 | 0.833 |
| Vision | screenshot available | 314 | 0.953 | 0.000 | 0.902 |
| Fusion | unrestricted | 700 | 1.000 | 1.000 | 1.000 |

Read this carefully, because the headline is misleading:

- The **URL branch alone is already perfect** on this test set. The fused model's
  1.000 is inherited, not earned by combining modalities. There is no measurable
  gain from HTML or vision.
- The **standalone HTML and vision models are not usable detectors**: both have
  near-zero specificity (0.00-0.36), i.e. they label most legitimate pages as
  phishing. Their F1 looks respectable only because the phishing class is ~92% of
  each availability-restricted subset.
- Their ROC-AUC (0.79 HTML, 0.90 vision) shows they carry *some* ranking signal,
  but nowhere near what their F1 suggests.

The honest summary: **on PhiUSIIL, URL string features saturate the task, and
page content contributes nothing measurable on top.** The multimodal pipeline is
built, tested, and explainable, but this dataset cannot demonstrate that it
helps.

## 1b. Modality availability is a label giveaway

The most important finding of this project is not a performance number. It is
that the multimodal benchmark is confounded.


| class | HTML available | screenshot available |
| --- | --- | --- |
| phishing (label 1) | 76.4% | 73.8% |
| legitimate (label 0) | 9.1% | 9.1% |

Availability gap: **+0.67**. A classifier that predicts "phishing" whenever HTML
was collectable, and never opens a page, reaches:

- accuracy 0.826
- precision 0.918
- recall 0.764
- **F1 0.834**

So any HTML, vision, or fusion number has to be read against **F1 0.834**, not
against 0.5. Evidence: `reports/availability_confound.md`.

The cause is the benchmark, not the collector. PhiUSIIL was labelled around
2018-2020; most phishing hosts in it have since expired, while most legitimate
sites are still online. Availability is therefore a proxy for "is this domain
still alive", which correlates with the label for reasons that have nothing to do
with phishing content.

**This is not fixable by collecting more pages from the same URLs.** It would need
a dataset of currently-live phishing pages (fresh captures, dated labels).

## 1c. The weak models misfire on real sites, live

A live `/analyze` of `https://www.wikipedia.org/` (a legitimate site):

| modality | probability of phishing | fusion weight |
| --- | ---: | ---: |
| URL | 0.00009 | 0.862 |
| HTML | 0.929 | 0.074 |
| Vision | 0.944 | 0.064 |
| **fused verdict** | **0.373** | |

The HTML and vision models call Wikipedia phishing with >92 % confidence. The
fused model only returns a correct verdict because the URL branch dominates the
gating. That is the specificity failure in §1a showing up on a real, obviously
legitimate page — not a theoretical concern.

Separately, the URL model alone classified `https://www.bbc.co.uk/news` as
phishing at p = 0.885, a false positive. Both were reproduced against the live
API, not inferred from the offline metrics.

The practical conclusion: **this system should not be deployed to make decisions
about individual URLs on the strength of these models.** The HTML and vision
branches would generate heavy false positives on the live web.

## 2. Scale of the multimodal evaluation

2,100 URLs were attempted; 970 had both modalities. That is enough to train a
small model and far too small to support confident claims about deep learning on
web content. Treat the multimodal numbers as a feasibility demonstration, not a
result.

The URL model is the one trained at meaningful scale (164,760 training rows).

## 3. What was never run

- **The `full` training profile was never executed.** It is configured, but
  everything reported here uses `dev` (40,000 train rows, ResNet-18). The
  estimate for a full-profile epoch on this CPU-only host was ~161 minutes.
- **The containerised screenshot service was never built.** Docker is not
  installed. Capture ran in-process with reduced isolation — no filesystem, PID,
  or network namespace around the browser. `/health` reports
  `screenshot_service: "in_process"` so the weaker posture is visible rather than
  implied. See `docs/security.md` §4.
- **DeiT-tiny was never trained**; the dev profile's ResNet-18 was used.
- **Pretrained backbone weights may not have loaded.** Whether they did is
  recorded per run in the checkpoint and the metrics report
  (`pretrained_loaded`). If that flag is false, the vision model is randomly
  initialised and must not be described as transfer-learned.

## 4. The URL model result is close to a ceiling, and the ceiling is suspicious

Test recall 0.9999, F1 0.9995, ROC-AUC 0.9998. That is not evidence of a strong
detector; it is evidence that the task as posed in PhiUSIIL is close to
separable by URL string features. Legitimate and phishing URLs in this dataset
differ in obvious ways (TLD mix, path structure, digit and entropy patterns).

The leakage experiment supports this reading: giving the model the engineered
label-adjacent columns *did not* improve recall (delta -0.00015). There was no
hidden shortcut to exploit because the standard features were already near
saturating.

A 0.999 F1 on a public benchmark should be read as **"this dataset is easy"**,
not "this model is excellent". Real-world phishing URLs overlap heavily with
legitimate ones, and no such separation exists in live traffic.

## 5. Threats to validity that remain

| Threat | Status |
| --- | --- |
| Domain leakage across splits | Mitigated — splits are grouped by registered domain, verified zero overlap |
| Label leakage via engineered features | Tested, not found (see leakage report) |
| Availability leakage | **Not mitigated — measured and reported instead** |
| Temporal drift | Not addressed; labels are ~6 years old |
| Live-traffic generalisation | Untested; would need a fresh, dated dataset |
| Calibration under the availability confound | Not established; calibrated probabilities are still indexed to this dataset's prior |

## 6. Operational limits

- **No authentication**, by requirement. Anyone who can reach the service can
  make it fetch arbitrary URLs on the server's network. The SSRF guard and rate
  limit are the only controls. A reverse proxy is required before exposure.
- The rate limiter is in-process and does not coordinate across workers.
- DNS rebinding between validation and connection is mitigated, not closed; see
  `docs/security.md` §2.
- Collection is slow and fragile against the open internet: dead domains, TLS
  failures, and 403s mean roughly half the attempted URLs yield nothing. Real
  deployments should source from a curated, dated capture set.

## 7. What would make the multimodal part of this project meaningful

1. A fresh, dated phishing dataset with currently-live pages, so availability
   stops predicting the label.
2. Evaluation on rows where the *mask is held constant* — compare models on the
   both-modalities-available subset, so liveness cannot leak into the score.
3. Collection at 10x this scale, with the containerised renderer for isolation.
4. Temporal holdout: train on older labels, test on newer captures.
5. Reporting availability-matched metrics as the primary number, with
   unrestricted numbers as a secondary diagnostic.

Point 2 is the cheapest and most valuable change, and it is the one that should
be done first.
