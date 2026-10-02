# Availability confound

Rows: 2100

## Modality availability by class

| split | label | n | html_ok | shot_ok |
| --- | --- | --- | --- | --- |
| test | 0 | 299 | 0.094 | 0.094 |
| test | 1 | 401 | 0.756 | 0.713 |
| train | 0 | 299 | 0.064 | 0.064 |
| train | 1 | 401 | 0.771 | 0.756 |
| val | 0 | 299 | 0.117 | 0.117 |
| val | 1 | 401 | 0.766 | 0.746 |

Availability gap (phishing - legitimate): **+0.672**

## Baseline: predict phishing iff HTML was collectable

| metric | value |
| --- | --- |
| accuracy | 0.8257 |
| precision | 0.9181 |
| recall | 0.7639 |
| f1 | 0.8339 |

## Reading

A model that never opens a page and only reads the availability mask already reaches F1 0.834. Any HTML, vision, or fusion model must be compared against this number, not against 0.5. A result near it means the model is reading domain liveness, not phishing content.

Cause: the dataset labels are from 2018-2020 and most phishing hosts in it have since expired, while most legitimate sites are still live. This is a property of the benchmark, not a bug in the collector, and it cannot be fixed by collecting more pages from the same URLs.
