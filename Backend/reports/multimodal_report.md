# Multimodal training report

Generated 2026-10-01T13:53:11.860318+00:00 by `training/train_multimodal.py`.

## Read this before comparing any number here

The URL branch in P2 trains on 164,760 rows (40,000 in the dev profile).
The HTML and vision branches can only train on pages that were **still
reachable in 2026** from a dataset crawled in 2022, which is a few hundred
rows, not hundreds of thousands. Metrics from these two branches are
therefore **not comparable** to the URL model's numbers, and are not
comparable to published PhiUSIIL results either.

They are reported because the system genuinely uses those modalities, and
because hiding them would misrepresent what the fusion network has learned.

Rows available: {'train': 700, 'val': 700, 'test': 700}

## HTML modality

> These numbers are computed **only on rows where HTML was collectable**, because the model cannot run on a missing page. Unrestricted scores, and the availability-matched comparison, are in `availability_controlled_metrics.md`. Phishing is ~92% of this subset, which is why F1 looks healthy while specificity is low.

- Training rows: **324**, validation rows: 339, test rows: **325**

| Metric | Value |
| --- | ---: |
| recall | 0.9700 |
| precision | 0.9479 |
| f1 | 0.9588 |
| specificity | 0.3600 |
| roc_auc | 0.7893 |
| pr_auc | 0.9556 |
| mcc | 0.3844 |
| brier | 0.0582 |

## Vision modality

> Computed **only on rows where a screenshot was captured**, for the same reason. See `availability_controlled_metrics.md`.

- Training rows: **322**, validation rows: 334, test rows: **314**

| Metric | Value |
| --- | ---: |
| recall | 1.0000 |
| precision | 0.9108 |
| f1 | 0.9533 |
| specificity | 0.0000 |
| roc_auc | 0.9020 |
| pr_auc | 0.9891 |
| mcc | 0.0000 |
| brier | 0.0674 |

## Fused model

- Test rows: **700**

| Metric | Value |
| --- | ---: |
| recall | 1.0000 |
| precision | 1.0000 |
| f1 | 1.0000 |
| specificity | 1.0000 |
| roc_auc | 1.0000 |
| pr_auc | 1.0000 |
| mcc | 1.0000 |
| brier | 0.0156 |

### Read this number correctly

A near-perfect fused score here is **not** evidence of a strong multimodal detector. Two separate things are being measured:

1. **The URL branch alone already scores 1.0 on this test set.** The fused model inherits that; page content adds nothing measurable on top of it. See `availability_controlled_metrics.md`.
2. **The availability mask is itself a label giveaway.** A phishing URL yields a page 76% of the time, a legitimate URL only 9%, because the phishing hosts have expired since labelling. A model reading only the mask already scores F1 0.834 (`availability_confound.md`).

The standalone HTML and vision models are the informative part, and they are weak: both have high recall but near-zero specificity, meaning they flag most legitimate pages as phishing.

### Learned modality weights (mean gate)

| Modality | Mean weight | Weight on phishing | Weight on legitimate |
| --- | ---: | ---: | ---: |
| URL | 0.8158 | 0.6876 | 0.9877 |
| HTML | 0.1166 | 0.1992 | 0.0057 |
| Vision | 0.0677 | 0.1132 | 0.0066 |

Modality availability on the test split: URL 100.0%, HTML 46.4%, Vision 44.9%.

### The missingness caveat, restated

Because availability differs by class, part of the fused model's signal
comes from *which modalities loaded*, not only from what they contain.
A deployment where every submitted URL resolves would not enjoy that
advantage. This is stated here rather than buried, and it is why the
gates above are reported per class.
