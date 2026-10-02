# Notebooks

Jupyter walkthrough of the training and preprocessing pipeline. Each notebook
**calls the production modules** in `../app/` and `../training/` rather than
reimplementing them, so a notebook can never drift from what the CLI scripts
and the API actually run.

## Run

All commands from `Backend/`:

```powershell
$py = 'E:\UIU\TRIMESTER - 11\Computer Security\CS_Project\Backend\.venv\Scripts\python.exe'
& $py -m jupyter lab --notebook-dir notebooks
```

Select the `Python 3` kernel. Everything else is already in the project venv.

## Order

| # | Notebook | Covers |
|---|---|---|
| 01 | `01_data_loading_and_inspection.ipynb` | Hash verification, column detection, label mapping, domain structure |
| 02 | `02_url_preprocessing.ipynb` | `CharTokenizer`, 30 handcrafted features, `FeatureScaler` |
| 03 | `03_url_model_training.ipynb` | CharCNN → BiLSTM → attention training loop |
| 04 | `04_evaluation_and_leakage.ipynb` | Metrics, bootstrap CIs, leakage check, mask-only baseline |
| 05 | `05_html_vision_preprocessing.ipynb` | 45 HTML features, HTML tokeniser, screenshot tensors |
| 06 | `06_multimodal_fusion_training.ipynb` | Frozen encoders, gated fusion, per-modality evaluation |

Notebook 03 is the slow one (CPU training). Notebook 06 uses random URL
embeddings as a placeholder so the fusion wiring is visible without depending on
a trained URL checkpoint.

## Traps these notebooks document

Each of these is a real behaviour in the code, verified while writing them.

- **Label convention.** `0 = Legitimate`, `1 = Phishing`. Any report claiming
  the reverse is wrong.
- **`use_handcrafted` silently disabled** when `n_handcrafted=0`
  (`url_model.py:86`). The 30 features are computed then discarded. Pass
  `n_handcrafted=30`.
- **`class_weights` is not a `pos_weight`.** It returns `[1.0, n_neg/n_pos]`, a
  per-class weight vector (`train_url.py:112`). Passing it to
  `BCEWithLogitsLoss(pos_weight=...)` is a shape error. Use `WeightedBCE`, which
  applies the weight per sample.
- **`build_rows` indentation bug.** Passing `tokenizer` with `scaler=None`
  replaces `url_mask` with a zero-width tensor (`multimodal_dataset.py:166`) —
  the `else` binds to the wrong `if`. Pass both or neither.
- **Checkpoint key names differ.** The CLI checkpoint uses `model_state` /
  `tokenizer` / `scaler`; notebook 03 writes `state_dict` / `tokenizer_state` /
  `scaler_state`. Notebooks accept either.
- **`HTMLTextTokenizer.fit` is not idempotent** — it appends to the vocabulary
  rather than rebuilding it. Call once.
- **`visible_text_of` is destructive** — it decomposes `<script>`/`<style>` in
  the soup you pass it.
- **Missing artifacts fail silently.** `_read_html` returns `None`,
  `_image_tensor` returns zeros. A deleted file becomes *mask says available,
  tensor is empty*. Notebooks 05 and 06 check this explicitly.
- **Availability is confounded with the label.** A mask-only classifier already
  scores F1 ~0.834 on this data. Any fusion gain must be reported against that
  baseline, not against chance. Notebook 06 demonstrates this directly: with
  untrained encoders and random URL embeddings, fusion still scores F1 1.0,
  because the mask alone predicts the label.

## Artifacts

Notebook outputs land in `notebooks/artifacts/`. They are regenerable and safe
to delete.

## Regenerating

The notebooks are produced by `_build_notebooks.py`, which keeps the shared
bootstrap cell consistent across all six:

```powershell
& $py notebooks/_build_notebooks.py
```

Edit the generator, not the `.ipynb` JSON, if you want a change to persist.

## Environment note

`jupyterlab`, `nbformat`, `nbconvert` and `ipykernel` were installed into the
project venv to support these notebooks. They are not in `requirements.txt`,
because the pipeline itself does not need them.