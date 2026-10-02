# P1 - Environment Summary

All values below were **measured on this machine**, not assumed. Commands are
quoted so every number can be re-derived.

- Host: `Windows-10-10.0.26300-SP0` (Windows 11 Home Single Language, build 26300)
- Collected: 2026-10-01 (UTC timestamps in the JSON artefacts are exact)

---

## 1. Python

| Item | Value | How measured |
| --- | --- | --- |
| Installed runtimes | 3.10.11, 3.13.14 | `py -0p` |
| **Selected** | **3.10.11** | matches the project target |
| Virtualenv | `backend/.venv` | created from 3.10.11 |
| Default interpreter on PATH | 3.10.11 (`...\Python310\python.exe`) | `where python` |

Python 3.11 is **not** installed on this host; the 3.10 target in the stack
matches what is actually available, so no interpreter work is needed.

## 2. CPU, RAM, GPU

| Item | Value |
| --- | --- |
| CPU | Intel(R) Core(TM) i5-1035G1 @ 1.00GHz |
| Cores / logical processors | 4 / 8 |
| torch threads used | 4 (`torch.get_num_threads()`) |
| RAM total | 11.81 GB |
| RAM free at P1 | ~1.4 GB |
| GPU 1 | NVIDIA GeForce MX330, 2 GB VRAM, driver 31.0.15.2866 |
| GPU 2 | Intel(R) UHD Graphics, 1 GB VRAM |
| `nvidia-smi` | driver 528.66, reports CUDA 12.0 |
| **`torch.cuda.is_available()`** | **`False`** |
| PyTorch build | `2.1.0+cpu` |

### The important consequence

A CUDA-capable NVIDIA driver is present, but the **installed PyTorch is the
CPU-only wheel**, so CUDA is unavailable to every training and inference
process:

```
>>> torch.cuda.is_available()
False
```

The MX330 is also a 2 GB entry-level part that would be marginal even with a
CUDA build. Consequently:

* Training runs on **CPU only**.
* `DEVICE=auto` correctly resolves to `cpu` (covered by a test).
* The `full` profile's DeiT-tiny vision training is **not practical** on this
  host. That is a hardware limitation, not a modelling choice, and it is why
  `dev` / `full` configs exist with identical architecture.

Free RAM is frequently under 2 GB, so `num_workers: 0` in the dev profile is a
deliberate choice: extra data-loader workers on this box increase memory
pressure and page faults rather than throughput.

## 3. Node / frontend

| Item | Value |
| --- | --- |
| Node.js | v22.22.3 |
| npm | 10.9.8 |
| `npm install` result | **added 251 packages, exit code 0** |

Every pinned version in `frontend/package.json` resolved and installed.

## 4. Docker

| Item | Value |
| --- | --- |
| `docker --version` | **not installed** (`CommandNotFoundException`) |

Consequences, stated plainly:

* `backend/docker/Dockerfile`, `backend/docker/screenshot-service.Dockerfile`
  and `backend/docker/docker-compose.yml` are written as the **deployment
  contract but have not been executed or verified** on this host.
* The isolated Chromium screenshot service required by the security design
  **cannot be built or tested here**. Until Docker is available, screenshot
  capture must be reported as *unavailable* rather than silently run unsandboxed
  on the host, which would violate the security requirements.
* This is a genuine blocker for P3's acceptance criterion and for P10's security
  review. It needs a decision from you.

## 5. Network

| Item | Value |
| --- | --- |
| PyPI reachable | yes (`pip download` and a full 84-package install succeeded) |
| npm registry reachable | yes (251 packages installed) |

## 6. Dependency verification

`backend/requirements.txt` pins every dependency with a one-line reason. All
pins were installed into `backend/.venv` and every key module imported
successfully:

```
>>> import torch, fastapi, sqlalchemy, captum, playwright, structlog, bs4, lxml, \
      tldextract, dns, sse_starlette, pytest, sklearn, pandas, numpy, yaml, \
      httpx, aiosqlite, slowapi
IMPORTS_OK
```

The fully resolved dependency graph (including transitive packages) is saved at
`reports/requirements_frozen.txt` - 84 packages.

Note: `dnspython` is installed as the PyPI distribution name; its import name is
`dns`.

## 7. Test status at the P1 gate

```
$ backend\.venv\Scripts\python.exe -m pytest -q
31 passed, 12 skipped in 12.69s
```

The 12 skips were all in `tests/test_splits.py` and were **expected at that
point**: they read the real split artefacts, which did not exist yet because the
dataset had not been located. They skip with an explicit message rather than
fabricating a fixture.

The full list of bugs found and fixed during P1 - including two that would have
produced silently wrong results - is in **section 10**.

## 8. Dataset

**PhiUSIIL was located** at
`Dataset/Phishing_URL_Dataset.csv` (project root, outside `backend/`).

It was **not** present when this phase began: the project directory was empty at
the start of the session and the initial recursive scan of `C:`, `D:` and `E:`
found nothing. The file's `LastWriteTime` is `2026-10-01 11:47:42`, i.e. it
appeared mid-session. It was found on a later sweep of the project directory, not
by the initial disk-wide search. The full measured profile is in
`reports/dataset_inspection.md`; the headline numbers:

| Field | Measured value |
| --- | --- |
| File | `Dataset/Phishing_URL_Dataset.csv` |
| Size | 54,160,642 bytes (51.65 MiB) |
| SHA-256 | `748ae7c7a677ac4ffc0eeefbee653dcef1d685b134ad2f5376596b92d79438ba` |
| Rows | 235,795 |
| Columns | 55 |
| URL column (detected) | `URL` (URL-parse rate 1.0) |
| Label column (detected) | `label`, values `{1: phishing, 0: legitimate}` |
| Class balance | 134,850 phishing (57.19 %) / 100,945 legitimate (42.81 %) |
| Duplicate full rows removed | 425 |
| Rows with unparseable host | 0 |
| Distinct registered domains | 175,510 |
| Domains carrying both labels | 120 |

The dataset is **not** committed: `.gitignore` excludes `backend/data/`. The file
currently sits in `Dataset/`, which is outside the ignore rules, so it **must be
either added to `.gitignore` or deliberately tracked as a data fixture** before
the first commit. No decision has been made on that yet.

### Data-quality notes (verified, not assumed)

* The file is **valid UTF-8**: 0 rows contain the Unicode replacement character
  U+FFFD. A first inspection through PowerShell's `Get-Content` appeared to show
  mojibake; that was a **console rendering artefact**, not corruption in the file.
* 126 `Title` values contain genuine Thai script. `Title` is a page-derived
  column and is excluded from the URL model regardless.
* 175,510 registered domains for 235,370 rows means the average domain
  contributes only ~1.34 rows. Grouped splitting therefore splits mostly
  single-row domains, which is the correct conservative behaviour here.
* 120 domains carry **both** labels. Grouped splitting keeps each such domain
  wholly inside one split; a row-level split would have leaked them.

### Other material found on this machine (not used)

| Path | What it is |
| --- | --- |
| `C:\Users\Asus\Downloads\PhishGuardHuge_model_files\` | two ~500 MB checkpoints from an unrelated project |
| `C:\Users\Asus\Downloads\demoProject\` | an empty Spring Boot Maven project |
| `E:\UIU\Project_Show\UIU_ML\MindCareAI\` | an unrelated existing ML project, left untouched |

I did **not** download PhiUSIIL from the internet. Its redistribution terms must
be reviewed before fetching, and the project rules require asking before adding
an external data source.

## 9. Split actually produced

`training/make_splits.py --seed 42` on the real file. Fingerprint
`514839cb599158da`.

| Split | Rows | Registered domains | Phishing | Legitimate | Phishing share |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 164,760 | 114,190 | 94,396 | 70,364 | 0.572930 |
| val | 35,305 | 29,760 | 20,227 | 15,078 | 0.572922 |
| test | 35,305 | 31,560 | 20,227 | 15,078 | 0.572922 |

Verified invariants, all from the script's own self-check:

```
"domain_overlap_counts": { "train|val": 0, "train|test": 0, "val|test": 0 }
"no_domain_overlap": true
"row_ids_total": 235370, "row_ids_unique": 235370, "row_ids_disjoint": true
```

Artefacts in `backend/data/splits/`: `train.txt`, `val.txt`, `test.txt`,
`domain_split.csv`, `dedup.csv`, `split_summary.json`.

## 10. Test status at the P1 gate

```
$ backend\.venv\Scripts\python.exe -m pytest -q
43 passed in 12.47s
```

All suites pass, including the 12 real-artifact split tests in
`tests/test_splits.py`, which read the actual split produced above.

### Real bugs the tests caught and fixed during P1

Found by tests or by running against the real file, not by inspection. They are
recorded because two of them would have produced silently wrong results:

1. **`make_splits` picked `IsDomainIP` as the label column.** Its heuristic was
   "first column with 2-5 distinct values", and `IsDomainIP` is a 0/1 feature
   column that appears before `label`. This produced a split that was
   **99.73 % single-class** (`label_counts {0: 164314, 1: 446}` on train) while
   the script still reported success. The split tool now delegates to the same
   detector the inspection tool uses.
2. **`inspect_dataset` also mis-detected the label** - as `HasFavicon` - for the
   same reason compounded by a 5-row head sample in which the real `label`
   column happened to contain only one class. Detection now counts the **whole**
   candidate column and prefers name matches, then the final column.
3. `pydantic-settings` JSON-parsed the tuple-typed settings out of `.env` and
   raised, so `ALLOWED_PORTS` / `ALLOWED_SCHEMES` / `ALLOWED_CONTENT_TYPES`
   could not be configured at all. Fixed with `NoDecode` annotations.
4. `structlog` was configured with `PrintLoggerFactory` while
   `add_logger_name` requires `logger.name`, so **every structured log record
   raised `AttributeError`**. Fixed with `structlog.stdlib.LoggerFactory`.
5. `stage_timer` overwrote the caller's own error context with the raised
   exception, losing the actual diagnosis. Both are now retained.
6. `grouped_split` ran a per-class greedy that could assign the *same domain* to
   two splits - the dict silently kept the last assignment, discarding the other
   and drifting the split ratio by up to 8 points. Rewritten as a single pass
   that respects overall and per-class targets.
7. `write_outputs` crashed with `TypeError` sorting dicts.
8. `run()` did not validate that split ratios sum to 1.0; only the CLI did.
9. `registered_domain_of` constructed a new `TLDExtract` per hostname. Over
   175k domains this made the split take ~4 minutes; caching it brings the run
   to ~59 s.

## 11. P1 acceptance criteria

| Criterion | Status |
| --- | --- |
| `reports/dataset_inspection.md` exists with real numbers | **MET** - generated from the real file, with SHA-256 provenance |
| Split files saved | **MET** - indices, domain map, dedup table and summary in `backend/data/splits/` |
| Split tests (no domain overlap) pass | **MET** - 12/12 pass against the real artefacts |

P1 acceptance criteria are satisfied. The Docker gap below is a P3 problem, not a
P1 one.

## 12. Open blocker carried forward

**Docker is not installed on this host.** The sandboxed Chromium screenshot
service cannot be built or verified here. Until Docker is available, screenshot
capture must be reported as *unavailable* rather than run unsandboxed on the
host. This affects P3's acceptance criteria and P10's security review, and needs
a decision.

---

# P2 - URL model

## 13. Training run actually executed

Command: `python training/train_url.py --config configs/dev.yaml --seed 42 --tag url`

| Field | Value |
| --- | --- |
| Device | `cpu` (no usable CUDA on this host) |
| Train / val / test rows | 40,000 / 35,305 / 35,305 |
| Train phishing rows | 22,917 |
| Tokenizer | 60 vocabulary entries, max length 249 (from the p99) |
| Handcrafted features | 30 |
| Epochs run | 0, 1, 2 then early stopping |
| Best epoch | 0 |
| Time per epoch | 816 - 914 s |

The dev profile caps training at 40,000 rows. The `full` profile uses the same
architecture on all 164,760 rows; at the measured 0.37 s/step that is roughly
161 minutes per epoch and was not run on this host.

## 14. Test-split results (real)

Platt scaling fitted on validation only. Brier 0.00053, ECE 0.00020.

| Metric | Value | 95 % bootstrap CI |
| --- | ---: | --- |
| Phishing recall | 0.9999011 | 0.9998001 - 1.0000000 |
| Precision | 0.9991602 | 0.9987675 - 0.9995559 |
| F1 | 0.9995305 | 0.9993549 - 0.9997286 |
| Specificity | 0.9988725 | - |
| ROC-AUC | 0.9998251 | - |
| PR-AUC | 0.9997782 | - |
| MCC | 0.9989005 | - |
| Brier | 0.0005331 | - |
| ECE | 0.0001966 | - |
| Recall @ 1 % FPR | 1.000 | 1.000 - 1.000 |

Calibration comparison on the test split:

| Method | Brier | ECE |
| --- | ---: | ---: |
| uncalibrated | 0.00057149 | 0.00262827 |
| temperature (T = 0.7070) | 0.00056957 | 0.00023993 |
| **platt (selected)** | **0.00053306** | **0.00019662** |
| isotonic | 0.00054535 | 0.00018760 |

## 15. Leakage experiment (real)

`python training/leakage_experiment.py --seed 42`, same test split and row ids.

| Model | Phishing recall | ROC-AUC | Brier |
| --- | ---: | ---: | ---: |
| URL-only (shipped) | 0.9999011 | 0.9998251 | 0.0005331 |
| engineered-column MLP | 0.9997528 | 0.9999900 | 0.0004408 |
| engineered logistic regression | 0.9997034 | 0.9999855 | 0.0007825 |

Delta recall (engineered − URL-only): **−0.0001483**. The page-derived columns
do **not** produce a better phishing-recall model, so the headline figure is not
inflated by leakage. Several individual page columns are near-separating on their
own (`URLSimilarityIndex` AUC 0.9936, `NoOfCSS` 0.9898, `LineOfCode` 0.9897),
which is precisely why the experiment exists. Full detail in
`reports/leakage_experiment.md`.

## 16. Real bugs the tests and verification caught in P2

1. **ECE returned `NaN`.** Empty calibration bins carry `gap = NaN`, and
   `0 * NaN` poisoned the sum. Caught by asserting ECE on a balanced set.
2. **Calibration was fitted on the wrong model's logits.** `train_url.py`
   reloaded the best checkpoint before test evaluation but reused the `val_logits`
   left over from the *last* epoch, so the calibrator was fitted against a
   different model than the one it was applied to. Caught because the fitted
   temperature (0.8325) matched neither a validation fit (0.7070) nor a test fit
   (0.7238); after the fix it reproduces the brute-force optimum exactly.
3. **The leakage experiment's feature set included the label.** The source file
   carries its own `label` column, which was being fed in as a feature. Now
   excluded, with a guard that aborts if any column reproduces the label exactly.
4. **`URLSimilarityIndex` was mislabelled as page-derived.** It is a
   corpus-level similarity count; the report now states that explicitly.
5. **`leakage_experiment.py` had no working entry point** (`raise SystemExit(0)`
   and no `main()`), so the experiment silently produced nothing while exiting 0.

## 17. Test status at the P2 gate

**189 tests pass** (`pytest`, ~19 s):

| File | Tests |
| --- | ---: |
| `test_url_model.py` | 48 |
| `test_configs.py` | 10 |
| `test_metrics.py` | 27 |
| `test_url_dataset.py` | 39 |
| `test_leakage_experiment.py` | 22 |
| split tests (P1) | 43 |