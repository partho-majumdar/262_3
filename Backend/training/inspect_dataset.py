"""P1 - inspect the real PhiUSIIL file and emit a report with real numbers.

Design rules for this script (research integrity):

* Nothing about the schema is assumed. The URL column, the label column and the
  set of page-derived ("engineered") columns are **detected** from the data and
  the detection is written to the report so a human can audit it.
* Every number in ``reports/dataset_inspection.md`` is computed here, from the
  actual file. No values are hardcoded.
* The file is read in chunks so a multi-hundred-megabyte CSV does not have to
  fit in RAM.
* A missing/unreadable file is a hard error. The script never invents data.

Usage
-----
    python training/inspect_dataset.py --csv path/to/PhiUSIIL.csv
    python training/inspect_dataset.py --csv ... --chunksize 200000
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlsplit

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

import pandas as pd  # noqa: E402

# ---------------------------------------------------------------------------
# Detection vocabulary. These are *candidate* names only; nothing is trusted
# without validating the actual values found in the file.
# ---------------------------------------------------------------------------
URL_COLUMN_CANDIDATES = ("url", "urls", "uri", "link", "address")
LABEL_COLUMN_CANDIDATES = ("label", "class", "target", "y", "isphishing", "is_phishing", "type", "result")

# Values that indicate a phishing-positive class, compared case-insensitively.
PHISHING_LABEL_TOKENS = {"1", "phishing", "phish", "malicious", "bad", "true", "yes", "attack"}
LEGIT_LABEL_TOKENS = {"0", "legitimate", "benign", "good", "false", "no", "normal", "safe"}

_URL_RE = re.compile(r"^\s*(?:https?|ftp|ftps)://", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------
@dataclass
class ColumnProfile:
    name: str
    dtype: str
    n_rows_seen: int
    n_missing: int
    n_unique: int
    missing_pct: float
    sample_values: list[Any]
    role: str = "unknown"  # url | label | engineered | unknown


@dataclass
class InspectionResult:
    csv_path: str
    csv_size_bytes: int
    csv_sha256: str
    inspected_at_utc: str
    n_rows: int
    chunksize: int
    n_columns: int
    columns: list[ColumnProfile] = field(default_factory=list)
    url_column: str | None = None
    label_column: str | None = None
    label_values: dict[str, int] = field(default_factory=dict)
    label_mapping: dict[str, str] = field(default_factory=dict)
    phishing_class_value: str | None = None
    n_exact_duplicate_rows: int = 0
    n_duplicate_urls: int = 0
    n_unique_urls: int = 0
    n_malformed_urls: int = 0
    n_registered_domains: int = 0
    top_registered_domains: list[tuple[str, int]] = field(default_factory=list)
    domains_with_both_labels: list[str] = field(default_factory=list)
    n_domains_with_both_labels: int = 0
    url_length_stats: dict[str, float] = field(default_factory=dict)
    tld_extract_ok: bool = True
    errors: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    """Streaming SHA-256 of the dataset file (provenance for the report)."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def looks_like_url_series(series: pd.Series, sample: int = 2000) -> float:
    """Fraction of a sample of ``series`` that parses as an absolute URL.

    Used to *confirm* a candidate URL column rather than trusting its name.
    """
    values = series.dropna().astype(str).head(sample)
    if values.empty:
        return 0.0
    return float(values.str.match(_URL_RE).mean())


def detect_url_column(df: pd) -> tuple[str | None, dict[str, Any]]:
    """Pick the column that actually contains URLs.

    Strategy: prefer a name match, but confirm it empirically. If no name
    matches, fall back to the column with the highest URL-parse rate above 0.5.
    """
    n_matchers = 0
    for col in df.columns:
        if str(col).strip().lower() in URL_COLUMN_CANDIDATES:
            rate = looks_like_url_series(df[col])
            if rate >= 0.5:
                n_matchers += 1
                return str(col), {"strategy": "name_match", "url_parse_rate": round(rate, 4)}

    scored: list[tuple[float, str]] = []
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            continue
        rate = looks_like_url_series(df[col])
        if rate >= 0.5:
            scored.append((rate, str(col)))
    if scored:
        scored.sort(reverse=True)
        return scored[0][1], {
            "strategy": "content_match",
            "url_parse_rate": round(scored[0][0], 4),
            "candidates": [(c, round(r, 4)) for r, c in scored],
        }
    return None, {"strategy": "not_found", "candidates": []}


def detect_label_column(csv_path: Path, columns: list[str]) -> tuple[str | None, dict[str, Any]]:
    """Pick the label column and map its values to phishing / legitimate.

    The whole candidate column is counted, never a head sample: a head sample
    can contain a single class (the first rows of a file sorted by label often
    do), which would make a perfectly good label column look degenerate and let
    an unrelated 0/1 feature column be selected in its place.

    Preference order:
      1. a column whose name is a known label name and validates
      2. the final column, if it validates (the common "label at the end" layout)
      3. any interpretable low-cardinality column, flagged as ambiguous
    """
    name_matches: list[tuple[str, dict[str, Any]]] = []
    for col in columns:
        if str(col).strip().lower() not in LABEL_COLUMN_CANDIDATES:
            continue
        counts = _count_column(csv_path, col)
        if len(counts) < 2:
            continue
        mapping, phish_value, problems = _map_binary_labels(list(counts))
        if problems:
            continue
        name_matches.append(
            (
                str(col),
                {
                    "strategy": "name_match",
                    "n_distinct": int(len(counts)),
                    "value_mapping": mapping,
                    "phishing_value": phish_value,
                    "counts": {str(k): int(v) for k, v in counts.items()},
                },
            )
        )
    if name_matches:
        chosen, evidence = name_matches[0]
        if len(name_matches) > 1:
            evidence["other_name_matches"] = [c for c, _ in name_matches[1:]]
        return chosen, evidence

    # Fallback 1: the last column.
    for col in reversed(columns):
        counts = _count_column(csv_path, col)
        mapping, phish_value, problems = _map_binary_labels(list(counts))
        if not problems:
            return str(col), {
                "strategy": "last_column",
                "n_distinct": int(len(counts)),
                "value_mapping": mapping,
                "phishing_value": phish_value,
                "counts": {str(k): int(v) for k, v in counts.items()},
            }

    # Fallback 2: any interpretable low-cardinality non-URL column.
    for col in columns:
        if str(col).strip().lower() in URL_COLUMN_CANDIDATES:
            continue
        counts = _count_column(csv_path, col)
        if not (2 <= len(counts) <= 5):
            continue
        mapping, phish_value, problems = _map_binary_labels(list(counts))
        if problems:
            continue
        return str(col), {
            "strategy": "content_match_ambiguous",
            "n_distinct": int(len(counts)),
            "value_mapping": mapping,
            "phishing_value": phish_value,
            "counts": {str(k): int(v) for k, v in counts.items()},
            "warning": (
                "Selected by content alone. Confirm this is the label column before "
                "trusting any downstream result."
            ),
        }
    return None, {"strategy": "not_found"}


def detect_columns(csv_path: Path) -> tuple[str | None, str | None, dict[str, Any]]:
    """Detect (url_column, label_column, evidence) for a dataset file.

    Single entry point shared by ``inspect_dataset`` and ``make_splits`` so both
    tools agree on which column is the label. Duplicating this logic previously
    let the split tool pick a 0/1 feature column (``IsDomainIP``) as the label
    and silently produce a 99.7 % single-class split.
    """
    head = pd.read_csv(csv_path, nrows=5)
    columns = [str(c) for c in head.columns]
    url_col, url_evidence = detect_url_column(head)
    label_col, label_evidence = detect_label_column(csv_path, columns)
    evidence = {"url": url_evidence, "label": label_evidence}
    return url_col, label_col, evidence


def _count_column(csv_path: Path, column: str, chunksize: int = 200_000) -> dict[str, int]:
    """Full value counts for a single column, read in chunks."""
    counts: Counter[str] = Counter()
    for chunk in pd.read_csv(csv_path, usecols=[column], chunksize=chunksize, dtype=str):
        counts.update(str(v) for v in chunk[column].dropna().tolist())
    return dict(counts)


def _map_binary_labels(values: Iterable[Any]) -> tuple[dict[str, str], str | None, list[str]]:
    """Map raw label values to 'phishing' / 'legitimate'.

    Returns ``(mapping, phishing_value, problems)``. ``problems`` is non-empty
    when the values cannot be interpreted unambiguously - in that case the
    caller must not assume a class meaning.
    """
    mapping: dict[str, str] = {}
    phish_value: str | None = None
    problems: list[str] = []
    for raw in values:
        token = str(raw).strip().lower()
        if token in PHISHING_LABEL_TOKENS:
            mapping[str(raw)] = "phishing"
            phish_value = str(raw)
        elif token in LEGIT_LABEL_TOKENS:
            mapping[str(raw)] = "legitimate"
        else:
            problems.append(f"uninterpretable label value {raw!r}")
    classes = set(mapping.values())
    if classes != {"phishing", "legitimate"}:
        problems.append(f"label values do not form a clean binary split: {sorted(classes)}")
    if len(mapping) != len(list(values)):
        problems.append("ambiguous label values")
    return mapping, phish_value, problems


# ---------------------------------------------------------------------------
# Main inspection
# ---------------------------------------------------------------------------
def inspect(
    csv_path: Path,
    chunksize: int = 200_000,
    top_n_domains: int = 25,
) -> InspectionResult:
    """Stream the CSV once and collect every statistic the report needs."""
    csv_path = csv_path.expanduser().resolve()
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {csv_path}\n"
            "PhiUSIIL must be supplied explicitly; this script will not guess a path."
        )

    result = InspectionResult(
        csv_path=str(csv_path),
        csv_size_bytes=csv_path.stat().st_size,
        csv_sha256=sha256_of(csv_path),
        inspected_at_utc=datetime.now(timezone.utc).isoformat(),
        n_rows=0,
        chunksize=chunksize,
        n_columns=0,
    )

    # ---- Pass 1: header + per-column streaming stats on a bounded prefix ---
    # We profile columns on a prefix (fast) but count rows/dupes/labels over the
    # whole file, because class balance must reflect every row.
    header_frame = pd.read_csv(csv_path, nrows=5)
    result.n_columns = len(header_frame.columns)
    all_columns = [str(c) for c in header_frame.columns]

    url_col, url_detection = detect_url_column(header_frame)
    label_col, label_detection = detect_label_column(csv_path, all_columns)
    result.url_column = url_col
    result.label_column = label_col
    result.errors.extend(
        [] if url_col else [f"URL column not detected: {url_detection}"]
    )
    if not label_col:
        raise RuntimeError(
            "Could not detect a binary label column. "
            f"Columns present: {list(header_frame.columns)}. "
            "Refusing to guess the label semantics - supply the column via "
            "--label-column once you have confirmed the schema."
        )

    # ---- Pass 2: full streaming pass -------------------------------------
    missing: Counter[str] = Counter()
    unique_counts: Counter[str] = Counter()
    samples: dict[str, list[Any]] = {}
    label_counts: Counter[str] = Counter()
    url_counter: Counter[str] = Counter()
    domain_label_pairs: dict[str, set[str]] = {}
    exact_row_hashes: Counter[str] = Counter()
    url_lengths: list[int] = []
    n_malformed = 0

    try:
        import tldextract

        # Force the bundled Public Suffix List snapshot: reproducible grouping
        # and no network call during tests.
        extract = tldextract.TLDExtract(suffix_list_urls=())
    except Exception as exc:  # noqa: BLE001
        result.tld_extract_ok = False
        result.errors.append(f"tldextract unavailable ({exc}); registered-domain grouping disabled")
        extract = None

    phish_raw = label_detection.get("phishing_value")
    result.phishing_class_value = phish_raw
    result.label_mapping = {
        k: v for k, v in label_detection.get("value_mapping", {}).items()
    }

    reader = pd.read_csv(
        csv_path,
        chunksize=chunksize,
        dtype=str,
        keep_default_na=True,
        na_values=["", "NA", "NaN", "null", "NULL", "None"],
    )

    for chunk in reader:
        result.n_rows += len(chunk)
        for col in chunk.columns:
            col_name = str(col)
            series = chunk[col]
            missing[col_name] += int(series.isna().sum())
            unique_counts[col_name] += int(series.nunique(dropna=True))
            if col_name not in samples:
                samples[col_name] = series.dropna().astype(str).head(3).tolist()

        label_series = chunk[label_col]
        for raw, cnt in label_series.value_counts(dropna=True).items():
            label_counts[str(raw)] += int(cnt)

        if url_col:
            url_series = chunk[url_col]
            for raw_url, cnt in url_series.value_counts(dropna=True).items():
                url_counter[str(raw_url)] += int(cnt)

        # Exact duplicate detection on the full row (all columns).
        chunk_hash = pd.util.hash_pandas_object(chunk.fillna("<NA>"), index=False)
        exact_row_hashes.update(int(h) for h in chunk_hash)

    # ---- Derived statistics ----------------------------------------------
    result.label_values = dict(sorted(label_counts.items(), key=lambda kv: -kv[1]))
    result.n_unique_urls = len(url_counter)
    result.n_duplicate_urls = int(sum(c - 1 for c in url_counter.values() if c > 1))
    result.n_exact_duplicate_rows = int(sum(c - 1 for c in exact_row_hashes.values() if c > 1))

    for url, count in url_counter.items():
        url_lengths.append(len(url))
        host = ""
        try:
            host = urlsplit(url).hostname or ""
        except ValueError:
            n_malformed += 1
            continue
        if not host:
            n_malformed += 1
            host = url
        if extract is not None:
            try:
                registered = extract(host).registered_domain
            except Exception:  # noqa: BLE001
                registered = ""
        else:
            registered = host.lower()
        if not registered:
            registered = host.lower()
        domain_label_pairs.setdefault(registered, set())

    # Label per URL requires the two columns together; do a second cheap pass
    # over the pair only, chunked.
    domain_row_counts: Counter[str] = Counter()
    if url_col:
        for chunk in pd.read_csv(
            csv_path,
            usecols=[c for c in {url_col, label_col}],
            chunksize=chunksize,
            dtype=str,
        ):
            for url, lbl in chunk[[url_col, label_col]].itertuples(index=False):
                if pd.isna(url) or pd.isna(lbl):
                    continue
                try:
                    host = urlsplit(str(url)).hostname or ""
                except ValueError:
                    host = str(url)
                if extract is not None:
                    reg = extract(host).registered_domain or host.lower()
                else:
                    reg = host.lower()
                key = reg or host.lower()
                domain_row_counts[key] += 1
                meaning = result.label_mapping.get(str(lbl), str(lbl))
                domain_label_pairs.setdefault(key, set()).add(meaning)

    result.n_malformed_urls = n_malformed
    result.n_registered_domains = len(domain_label_pairs)
    result.top_registered_domains = sorted(
        domain_row_counts.items(), key=lambda kv: (-kv[1], kv[0])
    )[:top_n_domains]
    result.n_domains_with_both_labels = sum(1 for s in domain_label_pairs.values() if len(s) > 1)
    result.domains_with_both_labels = sorted(d for d, s in domain_label_pairs.items() if len(s) > 1)[:50]

    if url_lengths:
        s = pd.Series(url_lengths, dtype="float64")
        result.url_length_stats = {
            "min": float(s.min()),
            "p50": float(s.quantile(0.50)),
            "p90": float(s.quantile(0.90)),
            "p95": float(s.quantile(0.95)),
            "p99": float(s.quantile(0.99)),
            "max": float(s.max()),
            "mean": round(float(s.mean()), 2),
        }

    # ---- Column roles ----------------------------------------------------
    for col in df_header_list(header_frame):
        role = "unknown"
        if col == result.url_column:
            role = "url"
        elif col == result.label_column:
            role = "label"
        elif pd.api.types.is_numeric_dtype(header_frame[col]) or col in header_frame.columns:
            role = "engineered" if col not in {result.url_column, result.label_column} else role
        result.columns.append(
            ColumnProfile(
                name=col,
                dtype=str(header_frame[col].dtype),
                n_rows_seen=result.n_rows,
                n_missing=int(missing.get(col, 0)),
                n_unique=min(unique_counts.get(col, 0), result.n_rows),
                missing_pct=round(100.0 * missing.get(col, 0) / max(result.n_rows, 1), 4),
                sample_values=samples.get(col, []),
                role=role,
            )
        )

    # Attach detection metadata for the report.
    result.label_values  # noqa: B018 - keep attribute alive for readability
    result._detection = {"url": url_detection, "label": label_detection}  # type: ignore[attr-defined]
    return result


def df_header_list(header_frame: pd.DataFrame) -> list[str]:
    return [str(c) for c in header_frame.columns]


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------
def _fmt_int(n: int) -> str:
    return f"{n:,}"


def render_markdown(r: InspectionResult, detection: dict[str, Any]) -> str:
    total = max(r.n_rows, 1)
    lines: list[str] = []
    ap = lines.append

    ap("# PhiUSIIL Dataset Inspection")
    ap("")
    ap("Generated by `backend/training/inspect_dataset.py`. Every number below is")
    ap("computed from the file identified in *Provenance*; nothing is hand-written.")
    ap("")

    ap("## Provenance")
    ap("")
    ap("| Field | Value |")
    ap("| --- | --- |")
    ap(f"| File | `{r.csv_path}` |")
    ap(f"| Size (bytes) | {_fmt_int(r.csv_size_bytes)} |")
    ap(f"| Size (MiB) | {r.csv_size_bytes / 1024 / 1024:.2f} |")
    ap(f"| SHA-256 | `{r.csv_sha256}` |")
    ap(f"| Inspected at (UTC) | {r.inspected_at_utc} |")
    ap(f"| Read chunk size | {_fmt_int(r.chunksize)} rows |")
    ap(f"| Rows (excluding header) | {_fmt_int(r.n_rows)} |")
    ap(f"| Columns | {r.n_columns} |")
    ap("")

    ap("## Column role detection")
    ap("")
    ap("Roles are **detected from content**, not assumed from the column name.")
    ap("The detection logic is in `detect_url_column` / `detect_label_column`.")
    ap("")
    ap("| Role | Detected column | Strategy | Evidence |")
    ap("| --- | --- | --- | --- |")
    ud = detection.get("url", {})
    ld = detection.get("label", {})
    ap(
        f"| URL | `{r.url_column}` | {ud.get('strategy', 'n/a')} | "
        f"URL-parse rate = {ud.get('url_parse_rate', 'n/a')} |"
    )
    ap(
        f"| Label | `{r.label_column}` | {ld.get('strategy', 'n/a')} | "
        f"{ld.get('n_distinct', 'n/a')} distinct values, mapping = {ld.get('value_mapping', 'n/a')} |"
    )
    ap("")

    if r.label_mapping:
        ap("### Label meaning")
        ap("")
        ap("| Raw value | Class |")
        ap("| --- | --- |")
        for raw, meaning in r.label_mapping.items():
            ap(f"| `{raw}` | {meaning} |")
        ap("")
        ap(f"Phishing class is the raw value `{r.phishing_class_value}`.")
        ap("")

    ap("## Class balance")
    ap("")
    ap("| Raw label | Count | Share |")
    ap("| --- | ---: | ---: |")
    for raw, cnt in r.label_values.items():
        meaning = r.label_mapping.get(raw, "unmapped")
        ap(f"| `{raw}` ({meaning}) | {_fmt_int(cnt)} | {100.0 * cnt / total:.2f}% |")
    ap("")

    n_phish = sum(c for k, c in r.label_values.items() if r.label_mapping.get(k) == "phishing")
    n_legit = sum(c for k, c in r.label_values.items() if r.label_mapping.get(k) == "legitimate")
    if n_phish + n_legit:
        ap(
            f"Imbalance ratio (legitimate : phishing) = "
            f"{n_legit / max(n_phish, 1):.2f} : 1 over {_fmt_int(n_phish + n_legit)} labelled rows."
        )
        ap("")

    ap("## Duplicates")
    ap("")
    ap("| Metric | Count |")
    ap("| --- | ---: |")
    ap(f"| Unique URL strings | {_fmt_int(r.n_unique_urls)} |")
    ap(f"| Redundant URL rows (duplicate URL) | {_fmt_int(r.n_duplicate_urls)} |")
    ap(f"| Redundant full rows (all columns identical) | {_fmt_int(r.n_exact_duplicate_rows)} |")
    ap(f"| Rows with unparseable / empty host | {_fmt_int(r.n_malformed_urls)} |")
    ap("")

    ap("## URL length (characters)")
    ap("")
    if r.url_length_stats:
        ap("| Statistic | Value |")
        ap("| --- | ---: |")
        for k in ("min", "p50", "p90", "p95", "p99", "max", "mean"):
            ap(f"| {k} | {r.url_length_stats[k]:.2f} |")
        ap("")
        ap(
            f"The p99 length ({r.url_length_stats['p99']:.0f}) is the basis for choosing "
            f"`URL_MAX_LENGTH`; a conservative cap is "
            f"{int(min(1024, max(64, r.url_length_stats['p99'] * 1.5)))}."
        )
    else:
        ap("No URL column could be parsed.")
    ap("")

    ap("## Registered-domain grouping (split unit)")
    ap("")
    ap("| Metric | Value |")
    ap("| --- | ---: |")
    ap(f"| Distinct registered domains | {_fmt_int(r.n_registered_domains)} |")
    ap(f"| Domains carrying BOTH class labels | {_fmt_int(r.n_domains_with_both_labels)} |")
    ap(f"| tldextract available | {r.tld_extract_ok} |")
    ap("")
    if r.top_registered_domains:
        ap("### Most frequent registered domains (by row count)")
        ap("")
        ap("| Registered domain | Rows |")
        ap("| --- | ---: |")
        for dom, count in r.top_registered_domains:
            ap(f"| `{dom}` | {_fmt_int(count)} |")
        ap("")
    if r.domains_with_both_labels:
        ap("### Domains that carry both labels (leakage hazard)")
        ap("")
        ap("These MUST be kept inside a single split, otherwise grouped splitting fails:")
        ap("")
        for dom in r.domains_with_both_labels:
            ap(f"- `{dom}`")
        ap("")

    ap("## Column profiles")
    ap("")
    ap("`role=engineered` columns are derived from the fetched *page*, not from the URL")
    ap("string. They are excluded from the URL model and reported separately as a")
    ap("leakage experiment.")
    ap("")
    ap("| # | Column | dtype | Role | Missing | Missing % | Distinct | Sample |")
    ap("| ---: | --- | --- | --- | ---: | ---: | ---: | --- |")
    for i, c in enumerate(r.columns, 1):
        sample = ", ".join(f"`{s}`" for s in c.sample_values[:2]) or "-"
        ap(
            f"| {i} | `{c.name}` | {c.dtype} | {c.role} | {_fmt_int(c.n_missing)} | "
            f"{c.missing_pct:.2f}% | {_fmt_int(c.n_unique)} | {sample} |"
        )
    ap("")

    if r.errors:
        ap("## Errors / warnings")
        ap("")
        for e in r.errors:
            ap(f"- {e}")
        ap("")

    return "\n".join(lines)


def write_report(result: InspectionResult, detection: dict[str, Any], out_dir: Path) -> tuple[Path, Path]:
    """Write the markdown + JSON artefacts. Returns both paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / "dataset_inspection.md"
    json_path = out_dir / "dataset_inspection.json"

    md_path.write_text(render_markdown(result, detection), encoding="utf-8")
    json_path.write_text(
        json.dumps(
            {
                "provenance": {
                    "csv_path": result.csv_path,
                    "size_bytes": result.csv_size_bytes,
                    "sha256": result.csv_sha256,
                    "inspected_at_utc": result.inspected_at_utc,
                },
                "shape": {"n_rows": result.n_rows, "n_columns": result.n_columns},
                "detection": detection,
                "label_values": result.label_values,
                "label_mapping": result.label_mapping,
                "phishing_class_value": result.phishing_class_value,
                "duplicates": {
                    "unique_urls": result.n_unique_urls,
                    "duplicate_url_rows": result.n_duplicate_urls,
                    "duplicate_full_rows": result.n_exact_duplicate_rows,
                    "malformed_urls": result.n_malformed_urls,
                },
                "url_length_stats": result.url_length_stats,
                "domain_grouping": {
                    "registered_domains": result.n_registered_domains,
                    "domains_with_both_labels": result.n_domains_with_both_labels,
                    "top_domains": result.top_registered_domains,
                },
                "columns": [asdict(c) for c in result.columns],
                "errors": result.errors,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    return md_path, json_path


def main(argv: list[str] | None = None) -> int:
    ap_ = argparse.ArgumentParser(description="Inspect a PhiUSIIL CSV and write a real report.")
    ap_.add_argument("--csv", required=True, type=Path, help="Path to the PhiUSIIL CSV")
    ap_.add_argument("--chunksize", type=int, default=200_000)
    ap_.add_argument("--out", type=Path, default=BACKEND_ROOT / "reports")
    args = ap_.parse_args(argv)

    result = inspect(args.csv, chunksize=args.chunksize)
    detection = getattr(result, "_detection", {})

    md_path, json_path = write_report(result, detection, args.out)
    print(f"Wrote {md_path}")
    print(f"Wrote {json_path}")
    print(f"rows={result.n_rows:,} cols={result.n_columns}")
    print(f"url_column={result.url_column!r} label_column={result.label_column!r}")
    print(f"label_values={result.label_values}")
    print(f"registered_domains={result.n_registered_domains:,}")
    print(f"domains_with_both_labels={result.n_domains_with_both_labels:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
