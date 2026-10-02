# P3 - Multimodal acquisition report

Generated 2026-10-01T12:07:44.997613+00:00 by `training/collect_multimodal.py`.

## What was collected

- URLs attempted: **2100**
- **HTML** fetched: 47.7% overall (76.4% phishing / 9.1% legitimate); availability gap +67.3%
- **Screenshot** fetched: 46.2% overall (73.8% phishing / 9.1% legitimate); availability gap +64.7%
- Rows with **both** modalities: 970

## The missingness confound

PhiUSIIL was crawled in 2022, so phishing hosts have largely expired.
If phishing pages fail to load more often than legitimate ones, then
*availability alone* predicts the label, and a fusion model can score
well by reading the modality mask instead of the content.

This is measured above rather than assumed. A deployed detector really
does know whether a page loaded, so the mask is a legitimate input - but
it means the multimodal headline numbers partly reward link rot, and they
are **not** comparable to a deployment where every URL resolves. Any
comparison against the URL-only model must be read with this in mind.

## Isolation posture

Docker is **not installed on this host**, so the containerised screenshot
service described in the deployment contract was not built or verified.
Screenshots were captured in-process with Chromium hardened from the
inside: a fresh isolated browser context per page, all non-http(s)
schemes aborted at the request layer (blocking `file://` reads and
`ftp://`/`ws://` exfiltration), all permissions denied, JS dialogs
auto-dismissed, downloads refused, and a hard navigation timeout.

**What this does not provide:** no filesystem, PID or network namespace
around the browser. A Chromium vulnerability would run with the
collector's privileges. This is a genuine reduction in security posture
and the containerised service remains the only form suitable for
untrusted traffic.

## Artifacts

Stored under `backend/data/` which is git-ignored; no fetched HTML or
screenshot is committed.
